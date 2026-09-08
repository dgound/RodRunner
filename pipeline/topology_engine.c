/*
 * topology_engine.c — C implementation of the Yamada polynomial engine
 * for theta-curves and two-loop links.
 *
 * Replaces the Python yamada.py. Called from detect_topology.py via ctypes.
 *
 * Optimizations over the straight translation:
 *   - Laurent polynomials track exponent bounds; add/mul touch only the span.
 *   - graph_R splits into connected components (R is multiplicative over
 *     disjoint union) before deletion-contraction.
 *   - The 3^c skein loop is OpenMP-parallel with a per-thread memo table.
 *   - Memo entries are variable-length and live in a bump arena that is
 *     reset once per cell (no per-entry malloc/free).
 *   - No per-call 2 KB memsets in canonicalize / vertex counting.
 *   - A crossing cap makes pathological projections fail fast instead of
 *     running for hours.
 *
 * Compile: gcc -O3 -fopenmp -shared -fPIC -o topology_engine.so topology_engine.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

/* ========================================================================
 * Limits. A diagram with more than ~20 crossings is not computable by the
 * 3^c state sum anyway, so 32 is generous.
 * ======================================================================== */
#define MAX_CROSSINGS     32
#define MAX_STRANDS       16
#define MAX_POINTS        512
#define MAX_SEGMENTS      512
#define MAX_DIAGRAM_EDGES 128   /* 3 strands + 2 per crossing */
#define MAX_GRAPH_EDGES   128

/* ========================================================================
 * 1. Laurent polynomials in A with exponent-bounds tracking.
 *    c[e + OFFSET] is the coefficient of A^e.  Zero polynomial: lo > hi.
 *    Slots outside [lo, hi] are never read.
 * ======================================================================== */
#define MAX_EXP   128
#define POLY_SIZE (2 * MAX_EXP + 1)
#define OFFSET    MAX_EXP

typedef struct {
    int lo, hi;
    int c[POLY_SIZE];
} LPoly;

static int g_overflow = 0;   /* set if an exponent left [-MAX_EXP, MAX_EXP] */

static inline void poly_set_zero(LPoly *p) { p->lo = 1; p->hi = 0; }
static inline int  poly_is_zero(const LPoly *p) { return p->lo > p->hi; }

static inline void poly_set_const(LPoly *p, int v) {
    if (v == 0) { poly_set_zero(p); return; }
    p->lo = p->hi = 0;
    p->c[OFFSET] = v;
}

/* -sigma = -(A + 1 + A^-1) */
static inline void poly_set_neg_sigma(LPoly *p) {
    p->lo = -1; p->hi = 1;
    p->c[OFFSET - 1] = -1; p->c[OFFSET] = -1; p->c[OFFSET + 1] = -1;
}

static inline void poly_trim(LPoly *p) {
    while (p->lo <= p->hi && p->c[p->lo + OFFSET] == 0) p->lo++;
    while (p->lo <= p->hi && p->c[p->hi + OFFSET] == 0) p->hi--;
}

static inline void poly_copy(LPoly *dst, const LPoly *src) {
    dst->lo = src->lo; dst->hi = src->hi;
    if (src->lo <= src->hi)
        memcpy(dst->c + src->lo + OFFSET, src->c + src->lo + OFFSET,
               (size_t)(src->hi - src->lo + 1) * sizeof(int));
}

/* acc += t   (in place, safe) */
static void poly_add_inplace(LPoly *acc, const LPoly *t) {
    if (poly_is_zero(t)) return;
    if (poly_is_zero(acc)) { poly_copy(acc, t); return; }
    int lo = MIN(acc->lo, t->lo), hi = MAX(acc->hi, t->hi);
    for (int e = lo; e <= hi; e++) {
        int v = 0;
        if (e >= acc->lo && e <= acc->hi) v += acc->c[e + OFFSET];
        if (e >= t->lo   && e <= t->hi)   v += t->c[e + OFFSET];
        acc->c[e + OFFSET] = v;
    }
    acc->lo = lo; acc->hi = hi;
    poly_trim(acc);
}

/* out = a * b   (out must not alias a or b) */
static void poly_mul(const LPoly *a, const LPoly *b, LPoly *out) {
    if (poly_is_zero(a) || poly_is_zero(b)) { poly_set_zero(out); return; }
    int lo = a->lo + b->lo, hi = a->hi + b->hi;
    if (lo < -MAX_EXP || hi > MAX_EXP) {
        g_overflow = 1;
        lo = MAX(lo, -MAX_EXP); hi = MIN(hi, MAX_EXP);
    }
    for (int e = lo; e <= hi; e++) out->c[e + OFFSET] = 0;
    for (int i = a->lo; i <= a->hi; i++) {
        int ai = a->c[i + OFFSET];
        if (!ai) continue;
        for (int j = b->lo; j <= b->hi; j++) {
            int e = i + j;
            if (e < -MAX_EXP || e > MAX_EXP) continue;
            out->c[e + OFFSET] += ai * b->c[j + OFFSET];
        }
    }
    out->lo = lo; out->hi = hi;
    poly_trim(out);
}

/* out = coef * A^exp * p */
static void poly_scale_mono(const LPoly *p, int exp, int coef, LPoly *out) {
    if (poly_is_zero(p) || coef == 0) { poly_set_zero(out); return; }
    int lo = p->lo + exp, hi = p->hi + exp;
    if (lo < -MAX_EXP || hi > MAX_EXP) g_overflow = 1;
    int clo = MAX(lo, -MAX_EXP), chi = MIN(hi, MAX_EXP);
    for (int e = clo; e <= chi; e++)
        out->c[e + OFFSET] = coef * p->c[e - exp + OFFSET];
    out->lo = clo; out->hi = chi;
    poly_trim(out);
}

static inline void poly_negate_inplace(LPoly *p) {
    for (int e = p->lo; e <= p->hi; e++) p->c[e + OFFSET] = -p->c[e + OFFSET];
}

/* Canonical representative modulo (-A)^n: min exponent -> 0, sign fixed. */
static void poly_normalize(const LPoly *p, LPoly *out) {
    if (poly_is_zero(p)) { poly_set_zero(out); return; }
    int m = p->lo;
    int sign = ((-m) % 2 != 0) ? -1 : 1;
    poly_scale_mono(p, -m, sign, out);
}

/* ========================================================================
 * 2. Memo table: bump arena + chained hash, variable-length entries.
 *    One Memo per thread; reset once per cell.
 * ======================================================================== */

typedef struct { int u, v; } Edge;

typedef struct Chunk {
    struct Chunk *next;
    size_t cap, used;
    unsigned char data[];
} Chunk;

#define CHUNK_BYTES (4u << 20)   /* 4 MB */

typedef struct MemoEntry {
    struct MemoEntry *next;
    unsigned hash;
    int iso, n_edges, plo, phi;
    /* followed by Edge[n_edges], then int coeff[phi - plo + 1] */
} MemoEntry;

#define MEMO_BUCKETS 32749

typedef struct {
    MemoEntry **buckets;
    Chunk *head;
    long long hits, misses;
} Memo;

static void *arena_alloc(Memo *m, size_t size) {
    size = (size + 7u) & ~(size_t)7u;
    if (!m->head || m->head->used + size > m->head->cap) {
        size_t cap = MAX((size_t)CHUNK_BYTES, size);
        Chunk *ch = (Chunk *)malloc(sizeof(Chunk) + cap);
        ch->next = m->head; ch->cap = cap; ch->used = 0;
        m->head = ch;
    }
    void *p = m->head->data + m->head->used;
    m->head->used += size;
    return p;
}

static Memo *memo_new(void) {
    Memo *m = (Memo *)calloc(1, sizeof(Memo));
    m->buckets = (MemoEntry **)calloc(MEMO_BUCKETS, sizeof(MemoEntry *));
    return m;
}

static void memo_free(Memo *m) {
    if (!m) return;
    Chunk *ch = m->head;
    while (ch) { Chunk *n = ch->next; free(ch); ch = n; }
    free(m->buckets);
    free(m);
}

static inline Edge *entry_edges(MemoEntry *e) {
    return (Edge *)((unsigned char *)e + ((sizeof(MemoEntry) + 7u) & ~(size_t)7u));
}
static inline int *entry_coeff(MemoEntry *e) {
    return (int *)(entry_edges(e) + e->n_edges);
}

static unsigned graph_hash(int iso, const Edge *edges, int n) {
    unsigned h = (unsigned)iso * 2654435761u ^ (unsigned)n * 40503u;
    for (int i = 0; i < n; i++) {
        h ^= (unsigned)(edges[i].u * 31 + edges[i].v) * 2246822519u;
        h = (h << 13) | (h >> 19);
    }
    return h;
}

static MemoEntry *memo_get(Memo *m, unsigned h, int iso, const Edge *edges, int n) {
    for (MemoEntry *e = m->buckets[h % MEMO_BUCKETS]; e; e = e->next) {
        if (e->hash != h || e->iso != iso || e->n_edges != n) continue;
        if (memcmp(entry_edges(e), edges, (size_t)n * sizeof(Edge)) == 0) return e;
    }
    return NULL;
}

static void memo_put(Memo *m, unsigned h, int iso, const Edge *edges, int n,
                     const LPoly *val) {
    int span = poly_is_zero(val) ? 0 : (val->hi - val->lo + 1);
    size_t hdr = (sizeof(MemoEntry) + 7u) & ~(size_t)7u;
    size_t size = hdr + (size_t)n * sizeof(Edge) + (size_t)span * sizeof(int);
    MemoEntry *e = (MemoEntry *)arena_alloc(m, size);
    e->hash = h; e->iso = iso; e->n_edges = n;
    e->plo = val->lo; e->phi = val->hi;
    memcpy(entry_edges(e), edges, (size_t)n * sizeof(Edge));
    if (span) memcpy(entry_coeff(e), val->c + val->lo + OFFSET, (size_t)span * sizeof(int));
    e->next = m->buckets[h % MEMO_BUCKETS];
    m->buckets[h % MEMO_BUCKETS] = e;
}

static inline void entry_to_poly(const MemoEntry *e, LPoly *out) {
    out->lo = e->plo; out->hi = e->phi;
    if (e->plo <= e->phi)
        memcpy(out->c + e->plo + OFFSET, entry_coeff((MemoEntry *)e),
               (size_t)(e->phi - e->plo + 1) * sizeof(int));
}

/* ========================================================================
 * 3. graph_R — Yamada graph invariant of an abstract multigraph.
 *    R(G) = R(G-e) + R(G/e)   e non-loop
 *    R(G) = -sigma R(G-e)     e loop
 *    R(k isolated) = (-1)^k
 *    R(G1 ⊔ G2) = R(G1) R(G2)  (follows from the above by induction)
 * ======================================================================== */

static int edge_cmp(const void *a, const void *b) {
    const Edge *x = (const Edge *)a, *y = (const Edge *)b;
    if (x->u != y->u) return x->u - y->u;
    return x->v - y->v;
}

/* Sort + relabel by first appearance. Returns number of distinct vertices.
 * `out` receives normalized canonical edges (u <= v, sorted). */
static int canonicalize(const Edge *edges, int n, Edge *out) {
    int maxv = -1;
    for (int i = 0; i < n; i++) {
        out[i].u = MIN(edges[i].u, edges[i].v);
        out[i].v = MAX(edges[i].u, edges[i].v);
        if (out[i].v > maxv) maxv = out[i].v;
    }
    if (n == 0) return 0;
    qsort(out, (size_t)n, sizeof(Edge), edge_cmp);
    int map[maxv + 1];
    for (int i = 0; i <= maxv; i++) map[i] = -1;
    int next = 0;
    for (int i = 0; i < n; i++) {
        if (map[out[i].u] < 0) map[out[i].u] = next++;
        if (map[out[i].v] < 0) map[out[i].v] = next++;
        int a = map[out[i].u], b = map[out[i].v];
        out[i].u = MIN(a, b); out[i].v = MAX(a, b);
    }
    qsort(out, (size_t)n, sizeof(Edge), edge_cmp);
    return next;
}

static void graph_R(Memo *m, int num_v, const Edge *edges, int n, LPoly *out);

/* Union-find helpers over labels 0..nv-1 */
static inline int uf_find(int *par, int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

/* edges: canonical labels 0..nv-1, every label has degree >= 1. */
static void graph_R_canon(Memo *m, int iso, const Edge *edges, int n, int nv, LPoly *out) {
    if (n == 0) { poly_set_const(out, (iso % 2) ? -1 : 1); return; }

    /* --- strip loops: factor (-sigma)^loops --- */
    int n_loops = 0;
    Edge rest[MAX_GRAPH_EDGES];
    int nr = 0;
    for (int i = 0; i < n; i++) {
        if (edges[i].u == edges[i].v) n_loops++;
        else rest[nr++] = edges[i];
    }
    if (n_loops > 0) {
        LPoly sub, f, ns, tmp;
        graph_R(m, nv, rest, nr, &sub);
        poly_set_neg_sigma(&ns);
        poly_set_const(&f, 1);
        for (int i = 0; i < n_loops; i++) { poly_mul(&f, &ns, &tmp); poly_copy(&f, &tmp); }
        poly_mul(&f, &sub, out);
        if (iso % 2) poly_negate_inplace(out);
        return;
    }

    /* --- connected components: R is multiplicative --- */
    {
        int par[nv];
        for (int i = 0; i < nv; i++) par[i] = i;
        for (int i = 0; i < n; i++) {
            int a = uf_find(par, edges[i].u), b = uf_find(par, edges[i].v);
            if (a != b) par[a] = b;
        }
        int root_of[nv], n_comp = 0;
        int comp_id[nv];
        for (int i = 0; i < nv; i++) comp_id[i] = -1;
        for (int i = 0; i < nv; i++) {
            int r = uf_find(par, i);
            if (comp_id[r] < 0) { comp_id[r] = n_comp; root_of[n_comp] = r; n_comp++; }
        }
        if (n_comp > 1) {
            LPoly prod, sub, tmp;
            poly_set_const(&prod, 1);
            Edge ce[MAX_GRAPH_EDGES];
            for (int c = 0; c < n_comp; c++) {
                int nce = 0, nverts = 0;
                for (int i = 0; i < nv; i++) if (uf_find(par, i) == root_of[c]) nverts++;
                for (int i = 0; i < n; i++)
                    if (uf_find(par, edges[i].u) == root_of[c]) ce[nce++] = edges[i];
                graph_R(m, nverts, ce, nce, &sub);
                poly_mul(&prod, &sub, &tmp);
                poly_copy(&prod, &tmp);
            }
            poly_copy(out, &prod);
            if (iso % 2) poly_negate_inplace(out);
            return;
        }
    }

    /* --- deletion-contraction on the first edge --- */
    int u = edges[0].u, v = edges[0].v;
    LPoly R_del, R_con;
    graph_R(m, nv, edges + 1, n - 1, &R_del);
    Edge merged[MAX_GRAPH_EDGES];
    for (int i = 1; i < n; i++) {
        merged[i - 1].u = (edges[i].u == v) ? u : edges[i].u;
        merged[i - 1].v = (edges[i].v == v) ? u : edges[i].v;
    }
    graph_R(m, nv - 1, merged, n - 1, &R_con);
    poly_copy(out, &R_del);
    poly_add_inplace(out, &R_con);
    if (iso % 2) poly_negate_inplace(out);
}

static void graph_R(Memo *m, int num_v, const Edge *edges, int n, LPoly *out) {
    if (n > MAX_GRAPH_EDGES) n = MAX_GRAPH_EDGES;
    Edge key[MAX_GRAPH_EDGES];
    int nd = canonicalize(edges, n, key);
    int iso = num_v - nd;
    unsigned h = graph_hash(iso, key, n);
    MemoEntry *e = memo_get(m, h, iso, key, n);
    if (e) { m->hits++; entry_to_poly(e, out); return; }
    m->misses++;
    graph_R_canon(m, iso, key, n, nd, out);
    memo_put(m, h, iso, key, n, out);
}

/* ========================================================================
 * 4. Crossing detection: 2D segment-segment intersection with depth.
 * ======================================================================== */

typedef struct {
    int seg_i, seg_j;
    double ti, tj, zi, zj, px, py;
} CrossingRecord;

/* Returns crossing count, -1 if non-generic, -2 if more than cap. */
static int find_crossings(const double *seg, const int *meta, int n_seg,
                          CrossingRecord *out, int cap) {
    int count = 0;
    for (int i = 0; i < n_seg; i++) {
        const double *a = seg + i * 6;
        const int *mi = meta + i * 3;
        for (int j = i + 1; j < n_seg; j++) {
            const int *mj = meta + j * 3;
            if (mi[1] == mj[1] || mi[1] == mj[2] || mi[2] == mj[1] || mi[2] == mj[2]) continue;
            const double *b = seg + j * 6;
            double dax = a[3]-a[0], day = a[4]-a[1], dbx = b[3]-b[0], dby = b[4]-b[1];
            double den = dax*dby - day*dbx;
            if (fabs(den) < 1e-9) continue;
            double ox = b[0]-a[0], oy = b[1]-a[1];
            double ta = (ox*dby - oy*dbx) / den, tb = (ox*day - oy*dax) / den;
            if (ta > 1e-7 && ta < 1.0-1e-7 && tb > 1e-7 && tb < 1.0-1e-7) {
                double za = a[2]*(1.0-ta) + a[5]*ta, zb = b[2]*(1.0-tb) + b[5]*tb;
                if (fabs(za - zb) < 1e-9) return -1;
                if (count >= cap) return -2;
                CrossingRecord *r = &out[count++];
                r->seg_i = i; r->seg_j = j; r->ti = ta; r->tj = tb; r->zi = za; r->zj = zb;
                r->px = a[0] + dax*ta; r->py = a[1] + day*ta;
            } else if (ta >= -1e-7 && ta <= 1.0+1e-7 && tb >= -1e-7 && tb <= 1.0+1e-7) {
                return -1;
            }
        }
    }
    for (int i = 0; i < count; i++)
        for (int j = i + 1; j < count; j++) {
            double dx = out[i].px - out[j].px, dy = out[i].py - out[j].py;
            if (dx*dx + dy*dy < 1e-12) return -1;
        }
    return count;
}

/* ========================================================================
 * 5. Spatial graph, diagram building, skein resolution.
 * ======================================================================== */

typedef struct {
    int n_vertices;
    double vertices[MAX_POINTS * 3];
    int n_strands;
    int strand_start[MAX_STRANDS], strand_len[MAX_STRANDS];
    double points[MAX_POINTS * 3];
    int strand_endv[MAX_STRANDS * 2];
} SpatialGraph;

typedef struct { int is_vertex, id, port; } DiagramEndpoint;
typedef struct { DiagramEndpoint a, b; } DiagramEdge;

static int build_segments(const SpatialGraph *sg, double tol,
                          double *seg6, int *meta3, int *seg_loc2) {
    int total = 0;
    for (int si = 0; si < sg->n_strands; si++) total += sg->strand_len[si];
    int pid[MAX_POINTS];
    for (int i = 0; i < total; i++) pid[i] = i;
    for (int i = 0; i < total; i++) {
        const double *p = sg->points + i * 3;
        for (int j = i + 1; j < total; j++) {
            if (pid[j] != j) continue;
            double dx = p[0]-sg->points[j*3], dy = p[1]-sg->points[j*3+1], dz = p[2]-sg->points[j*3+2];
            if (dx*dx + dy*dy + dz*dz < tol*tol) pid[j] = pid[i];
        }
    }
    int n = 0;
    for (int si = 0; si < sg->n_strands; si++) {
        int st = sg->strand_start[si], len = sg->strand_len[si];
        for (int k = 0; k < len - 1 && n < MAX_SEGMENTS; k++) {
            const double *p0 = sg->points + (st+k)*3, *p1 = sg->points + (st+k+1)*3;
            memcpy(seg6 + n*6, p0, 3*sizeof(double));
            memcpy(seg6 + n*6 + 3, p1, 3*sizeof(double));
            meta3[n*3] = si; meta3[n*3+1] = pid[st+k]; meta3[n*3+2] = pid[st+k+1];
            seg_loc2[n*2] = si; seg_loc2[n*2+1] = k;
            n++;
        }
    }
    return n;
}

static void rotate_segments(const double *seg, double *out, int n, const double *R) {
    for (int i = 0; i < n * 2; i++) {
        const double *p = seg + i*3; double *q = out + i*3;
        q[0] = p[0]*R[0] + p[1]*R[1] + p[2]*R[2];
        q[1] = p[0]*R[3] + p[1]*R[4] + p[2]*R[5];
        q[2] = p[0]*R[6] + p[1]*R[7] + p[2]*R[8];
    }
}

static long long g_last_states = 0, g_last_hits = 0, g_last_misses = 0;

static void build_and_resolve(const SpatialGraph *sg, const CrossingRecord *x, int nx,
                              const double *rseg, const int *seg_loc, LPoly *total) {
    int nV = sg->n_vertices;

    /* CCW port assignment per crossing. label: 0=over_out 1=over_in 2=under_out 3=under_in */
    int slot[MAX_CROSSINGS][4];
    for (int k = 0; k < nx; k++) {
        const double *a = rseg + x[k].seg_i*6, *b = rseg + x[k].seg_j*6;
        int over_first = x[k].zi > x[k].zj;
        double di[2] = {a[3]-a[0], a[4]-a[1]}, dj[2] = {b[3]-b[0], b[4]-b[1]};
        double *dO = over_first ? di : dj, *dU = over_first ? dj : di;
        double dirs[4][2] = {{dO[0],dO[1]},{-dO[0],-dO[1]},{dU[0],dU[1]},{-dU[0],-dU[1]}};
        double ang[4]; int order[4] = {0,1,2,3};
        for (int i = 0; i < 4; i++) ang[i] = atan2(dirs[i][1], dirs[i][0]);
        for (int i = 0; i < 3; i++) for (int j = i+1; j < 4; j++)
            if (ang[order[i]] > ang[order[j]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
        while (order[0] >= 2) { int t = order[0]; order[0]=order[1]; order[1]=order[2]; order[2]=order[3]; order[3]=t; }
        for (int pos = 0; pos < 4; pos++) slot[k][order[pos]] = pos;
    }

    /* Per-strand ordered crossing events */
    typedef struct { int seg; double t; int k; int over; } Ev;
    Ev ev[MAX_STRANDS][MAX_CROSSINGS];
    int nev[MAX_STRANDS];
    memset(nev, 0, sizeof(nev));
    for (int k = 0; k < nx; k++) {
        int gi = x[k].seg_i, gj = x[k].seg_j, of = x[k].zi > x[k].zj;
        int si = seg_loc[gi*2], sj = seg_loc[gj*2];
        ev[si][nev[si]++] = (Ev){seg_loc[gi*2+1], x[k].ti, k, of ? 1 : 0};
        ev[sj][nev[sj]++] = (Ev){seg_loc[gj*2+1], x[k].tj, k, of ? 0 : 1};
    }
    for (int s = 0; s < sg->n_strands; s++)
        for (int i = 0; i < nev[s]-1; i++) for (int j = i+1; j < nev[s]; j++) {
            Ev *A = &ev[s][i], *B = &ev[s][j];
            if (A->seg > B->seg || (A->seg == B->seg && A->t > B->t)) { Ev t = *A; *A = *B; *B = t; }
        }

    /* Diagram edges */
    DiagramEdge de[MAX_DIAGRAM_EDGES];
    int nde = 0;
    for (int s = 0; s < sg->n_strands; s++) {
        DiagramEndpoint prev = {1, sg->strand_endv[s*2], 0};
        for (int i = 0; i < nev[s]; i++) {
            Ev *e = &ev[s][i];
            DiagramEndpoint cur = {0, e->k, e->over ? slot[e->k][1] : slot[e->k][3]};
            de[nde++] = (DiagramEdge){prev, cur};
            prev = (DiagramEndpoint){0, e->k, e->over ? slot[e->k][0] : slot[e->k][2]};
        }
        DiagramEndpoint end = {1, sg->strand_endv[s*2+1], 0};
        de[nde++] = (DiagramEdge){prev, end};
    }

    /* Skein sum over 3^nx states, parallel with per-thread memo */
    long long n_states = 1;
    for (int i = 0; i < nx; i++) n_states *= 3;
    g_last_states = n_states;
    g_last_hits = g_last_misses = 0;
    poly_set_zero(total);

    #pragma omp parallel if(n_states >= 729)
    {
        Memo *memo = memo_new();
        LPoly local; poly_set_zero(&local);

        #pragma omp for schedule(dynamic, 32)
        for (long long sid0 = 0; sid0 < n_states; sid0++) {
            int st[MAX_CROSSINGS];
            long long sid = sid0; int exp = 0;
            for (int i = 0; i < nx; i++) { st[i] = (int)(sid % 3); sid /= 3; exp += (st[i]==0) - (st[i]==1); }

            int xmap[MAX_CROSSINGS][4], next_id = nV;
            for (int k = 0; k < nx; k++) {
                if (st[k] == 2) { int v = next_id++; xmap[k][0]=xmap[k][1]=xmap[k][2]=xmap[k][3]=v; }
                else if (st[k] == 0) { int v1 = next_id++, v2 = next_id++; xmap[k][0]=xmap[k][3]=v1; xmap[k][1]=xmap[k][2]=v2; }
                else { int v1 = next_id++, v2 = next_id++; xmap[k][0]=xmap[k][1]=v1; xmap[k][2]=xmap[k][3]=v2; }
            }
            Edge g[MAX_DIAGRAM_EDGES];
            for (int i = 0; i < nde; i++) {
                g[i].u = de[i].a.is_vertex ? de[i].a.id : xmap[de[i].a.id][de[i].a.port];
                g[i].v = de[i].b.is_vertex ? de[i].b.id : xmap[de[i].b.id][de[i].b.port];
            }
            LPoly val, term;
            graph_R(memo, next_id, g, nde, &val);
            poly_scale_mono(&val, exp, 1, &term);
            poly_add_inplace(&local, &term);
        }

        #pragma omp critical
        {
            poly_add_inplace(total, &local);
            g_last_hits += memo->hits; g_last_misses += memo->misses;
        }
        memo_free(memo);
    }
}

/* ========================================================================
 * 6. Public API
 * ======================================================================== */

static void random_rotation(double *R, unsigned long long *s) {
    double M[9];
    for (int i = 0; i < 9; i++) {
        *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
        unsigned long long x = *s * 0x2545F4914F6CDD1DULL;
        M[i] = ((double)(x >> 11)) / (double)(1ULL << 53) * 2.0 - 1.0;
    }
    double *r0 = R, *r1 = R+3, *r2 = R+6, n, d;
    r0[0]=M[0]; r0[1]=M[1]; r0[2]=M[2];
    n = sqrt(r0[0]*r0[0]+r0[1]*r0[1]+r0[2]*r0[2]); if (n < 1e-10) { r0[0]=1; r0[1]=r0[2]=0; n=1; }
    r0[0]/=n; r0[1]/=n; r0[2]/=n;
    r1[0]=M[3]; r1[1]=M[4]; r1[2]=M[5];
    d = r1[0]*r0[0]+r1[1]*r0[1]+r1[2]*r0[2];
    r1[0]-=d*r0[0]; r1[1]-=d*r0[1]; r1[2]-=d*r0[2];
    n = sqrt(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2]); if (n < 1e-10) { r1[0]=0; r1[1]=1; r1[2]=0; n=1; }
    r1[0]/=n; r1[1]/=n; r1[2]/=n;
    r2[0]=r0[1]*r1[2]-r0[2]*r1[1]; r2[1]=r0[2]*r1[0]-r0[0]*r1[2]; r2[2]=r0[0]*r1[1]-r0[1]*r1[0];
}

/* Returns crossings in best projection (>= 0),
 * -1: no generic projection found, -2: every projection exceeds max_crossings,
 * -3: input too large for compiled limits. */
static int yamada_polynomial(const double *strands, const int *strand_lens, int n_strands,
                             double tol, int tries, unsigned long long seed, int max_crossings,
                             LPoly *result) {
    if (n_strands > MAX_STRANDS) return -3;
    if (max_crossings > MAX_CROSSINGS) max_crossings = MAX_CROSSINGS;

    SpatialGraph sg; sg.n_strands = n_strands;
    int total = 0;
    for (int si = 0; si < n_strands; si++) {
        if (total + strand_lens[si] > MAX_POINTS) return -3;
        sg.strand_start[si] = total; sg.strand_len[si] = strand_lens[si];
        memcpy(sg.points + total*3, strands + total*3, (size_t)strand_lens[si]*3*sizeof(double));
        total += strand_lens[si];
    }
    sg.n_vertices = 0;
    for (int si = 0; si < n_strands; si++) for (int ep = 0; ep < 2; ep++) {
        const double *p = sg.points + (sg.strand_start[si] + (ep ? sg.strand_len[si]-1 : 0))*3;
        int found = -1;
        for (int vi = 0; vi < sg.n_vertices; vi++) {
            double *q = sg.vertices + vi*3;
            double dx=p[0]-q[0], dy=p[1]-q[1], dz=p[2]-q[2];
            if (dx*dx+dy*dy+dz*dz < tol*tol) { found = vi; break; }
        }
        if (found < 0) { found = sg.n_vertices++; memcpy(sg.vertices + found*3, p, 3*sizeof(double)); }
        sg.strand_endv[si*2+ep] = found;
    }

    double seg6[MAX_SEGMENTS*6], rseg[MAX_SEGMENTS*6], best_rseg[MAX_SEGMENTS*6];
    int meta3[MAX_SEGMENTS*3], seg_loc2[MAX_SEGMENTS*2];
    int n_seg = build_segments(&sg, tol, seg6, meta3, seg_loc2);

    CrossingRecord xg[MAX_CROSSINGS], best_x[MAX_CROSSINGS];
    int best_nx = -1, saw_cap = 0;
    unsigned long long rng = seed ? seed : 42;

    for (int trial = 0; trial < tries; trial++) {
        double R[9] = {1,0,0, 0,1,0, 0,0,1};
        if (trial > 0) random_rotation(R, &rng);
        rotate_segments(seg6, rseg, n_seg, R);
        int cap = (best_nx < 0) ? max_crossings : MIN(best_nx, max_crossings);
        int nx = find_crossings(rseg, meta3, n_seg, xg, cap);
        if (nx == -2) { saw_cap = 1; continue; }
        if (nx < 0) continue;
        int bad = 0;
        for (int vi = 0; vi < sg.n_vertices && !bad; vi++) {
            double *v = sg.vertices + vi*3;
            double vx = v[0]*R[0]+v[1]*R[1]+v[2]*R[2], vy = v[0]*R[3]+v[1]*R[4]+v[2]*R[5];
            for (int k = 0; k < nx; k++) {
                double dx = vx-xg[k].px, dy = vy-xg[k].py;
                if (dx*dx+dy*dy < 1e-12) { bad = 1; break; }
            }
        }
        if (bad) continue;
        if (best_nx < 0 || nx < best_nx) {
            best_nx = nx;
            memcpy(best_x, xg, (size_t)nx*sizeof(CrossingRecord));
            memcpy(best_rseg, rseg, (size_t)n_seg*6*sizeof(double));
        }
        if (best_nx <= 2) break;
    }
    if (best_nx < 0) return saw_cap ? -2 : -1;

    build_and_resolve(&sg, best_x, best_nx, best_rseg, seg_loc2, result);
    return best_nx;
}

/* ---- ctypes entry points ---- */

int compute_yamada(const double *strand_points, const int *strand_lens, int n_strands,
                   double tol, int tries, unsigned long long seed, int max_crossings,
                   int *out_coeff, int *out_n_crossings) {
    LPoly poly, norm;
    g_overflow = 0;
    int ret = yamada_polynomial(strand_points, strand_lens, n_strands, tol, tries, seed,
                                max_crossings, &poly);
    memset(out_coeff, 0, POLY_SIZE * sizeof(int));
    *out_n_crossings = ret >= 0 ? ret : 0;
    if (ret < 0) return ret;
    poly_normalize(&poly, &norm);
    for (int e = norm.lo; e <= norm.hi; e++) out_coeff[e + OFFSET] = norm.c[e + OFFSET];
    return ret;
}

int  get_poly_size(void)      { return POLY_SIZE; }
int  get_offset(void)         { return OFFSET; }
int  get_max_crossings(void)  { return MAX_CROSSINGS; }
int  get_overflow_flag(void)  { return g_overflow; }
long long get_last_states(void) { return g_last_states; }
long long get_last_hits(void)   { return g_last_hits; }
long long get_last_misses(void) { return g_last_misses; }
int  has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
void set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}
/* Kept for API compatibility; memo is now per-call. */
void clear_memo(void) {}
