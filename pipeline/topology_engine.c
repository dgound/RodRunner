/*
 * topology_engine.c — C implementation of the Yamada polynomial engine
 * for theta-curves and two-loop links.
 *
 * Replaces the Python yamada.py, topology_core.py and topology_yamada.py.
 * Called from detect_topology.py via ctypes; also provides the theta-pattern
 * classifier, global sister assignment, strand/loop validation and the
 * classical linking number.
 *
 * Optimizations over the straight translation:
 *   - Laurent polynomials track exponent bounds; add/mul touch only the span.
 *   - graph_R applies exact reductions first: a degree-1 vertex forces R = 0
 *     and degree-2 vertices are contracted (subdivision invariance), so the
 *     deletion-contraction recursion never walks smoothing chains.
 *   - graph_R splits into connected components (R is multiplicative over
 *     disjoint union) before deletion-contraction.
 *   - The 3^c skein loop is OpenMP-parallel with a per-thread memo table.
 *   - Memo entries are variable-length and live in a bump arena that is
 *     cleared when a thread exceeds its memory budget (default 256 MB).
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
    size_t bytes;
    long long hits, misses, resets;
} Memo;

/* Per-thread memo budget; the table is cleared and its arena released when a
 * cell would exceed it, so peak memory is threads x budget rather than
 * unbounded in the number of skein states. */
static size_t g_memo_budget = (size_t)256 << 20;

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
    m->bytes += size;
    return p;
}

static void memo_reset(Memo *m) {
    Chunk *ch = m->head;
    while (ch) { Chunk *n = ch->next; free(ch); ch = n; }
    m->head = NULL;
    memset(m->buckets, 0, MEMO_BUCKETS * sizeof(MemoEntry *));
    m->bytes = 0;
    m->resets++;
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
    if (m->bytes + size > g_memo_budget) memo_reset(m);
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

/* Exact shortcuts before deletion-contraction:
 *   - a degree-1 vertex forces R(G) = 0:  R(G) = R(G-e) + R(G/e) and
 *     G-e = H + isolated vertex, so R(G) = -R(H) + R(H).
 *   - a degree-2 vertex on two distinct non-loop edges is a subdivision;
 *     R(G) = R(G/e) because the deletion term has a pendant edge.
 * Returns the number of vertices removed, or -1 if R(G) = 0. */
static int series_reduce(Edge *edges, int *n_io, int nd) {
    int n = *n_io;
    if (nd == 0) return 0;
    int deg[nd];
    for (int i = 0; i < nd; i++) deg[i] = 0;
    for (int i = 0; i < n; i++) { deg[edges[i].u]++; deg[edges[i].v]++; }
    for (int i = 0; i < nd; i++) if (deg[i] == 1) return -1;
    int removed = 0;
    for (int v = 0; v < nd; v++) {
        if (deg[v] != 2) continue;
        int first = -1, second = -1;
        for (int i = 0; i < n && second < 0; i++) {
            if (edges[i].u < 0 || edges[i].u == edges[i].v) continue;
            if (edges[i].u == v || edges[i].v == v) { if (first < 0) first = i; else second = i; }
        }
        if (second < 0) continue;            /* degree 2 via a single loop */
        int a = edges[first].u == v ? edges[first].v : edges[first].u;
        int b = edges[second].u == v ? edges[second].v : edges[second].u;
        edges[first].u = a; edges[first].v = b;
        edges[second].u = edges[second].v = -1;
        deg[v] = 0;
        removed++;
    }
    if (removed) {
        int k = 0;
        for (int i = 0; i < n; i++) if (edges[i].u >= 0) edges[k++] = edges[i];
        *n_io = k;
    }
    return removed;
}

static void graph_R(Memo *m, int num_v, const Edge *edges, int n, LPoly *out) {
    if (n > MAX_GRAPH_EDGES) n = MAX_GRAPH_EDGES;
    Edge key[MAX_GRAPH_EDGES];
    int nd = canonicalize(edges, n, key);
    int removed = series_reduce(key, &n, nd);
    if (removed < 0) { poly_set_zero(out); return; }
    if (removed > 0) {
        Edge reduced[MAX_GRAPH_EDGES];
        memcpy(reduced, key, (size_t)n * sizeof(Edge));
        nd = canonicalize(reduced, n, key);
        num_v -= removed;
    }
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

static long long g_last_states = 0, g_last_hits = 0, g_last_misses = 0, g_last_resets = 0;

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
    g_last_hits = g_last_misses = g_last_resets = 0;
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
            g_last_resets += memo->resets;
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

/* ========================================================================
 * 7. Sister assignment: globally optimal 2-state DP over unordered pairs.
 *    Mirrors topology_core.assign_sisters, including the best-two history
 *    bookkeeping and (cost, path) lexicographic tie-break.
 * ======================================================================== */

#define MAX_LOCI 64

typedef struct { double cost; unsigned char path[MAX_LOCI]; int len; } Hist;

static int path_cmp(const unsigned char *a, const unsigned char *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return (int)a[i] - (int)b[i];
    return 0;
}

/* Insert candidate into a best-two list ordered by (cost, path). */
static void best_two_insert(Hist *list, int *count, const Hist *cand) {
    for (int i = 0; i < *count; i++)
        if (list[i].len == cand->len && path_cmp(list[i].path, cand->path, cand->len) == 0) {
            if (cand->cost < list[i].cost) list[i].cost = cand->cost;
            goto resort;
        }
    if (*count < 2) list[(*count)++] = *cand;
    else {
        int worst = 0;
        if (list[1].cost > list[0].cost ||
            (list[1].cost == list[0].cost && path_cmp(list[1].path, list[0].path, list[1].len) > 0))
            worst = 1;
        int better = cand->cost < list[worst].cost ||
            (cand->cost == list[worst].cost && path_cmp(cand->path, list[worst].path, cand->len) < 0);
        if (better) list[worst] = *cand;
    }
resort:
    if (*count == 2) {
        int swap = list[1].cost < list[0].cost ||
            (list[1].cost == list[0].cost && path_cmp(list[1].path, list[0].path, list[1].len) < 0);
        if (swap) { Hist t = list[0]; list[0] = list[1]; list[1] = t; }
    }
}

static inline double dist_dims(const double *p, const double *q, int dim_mask) {
    double s = 0.0;
    for (int d = 0; d < 3; d++) if (dim_mask & (1 << d)) { double t = p[d] - q[d]; s += t * t; }
    return sqrt(s);
}

/* comp1/comp2: [n][3]. Oriented A/B: state 0 -> (c1, c2), state 1 -> (c2, c1). */
static double transition(const double *c1, const double *c2, int n, int i, int si,
                         int j, int sj, int dim_mask) {
    const double *la = si ? c2 + i*3 : c1 + i*3, *lb = si ? c1 + i*3 : c2 + i*3;
    const double *ra = sj ? c2 + j*3 : c1 + j*3, *rb = sj ? c1 + j*3 : c2 + j*3;
    (void)n;
    return dist_dims(la, ra, dim_mask) + dist_dims(lb, rb, dim_mask);
}

/* Returns 0 on success, -1 on bad input.
 * comp1/comp2 are sorted in place so comp1 <= comp2 lexicographically (as Python does).
 * runner_up is NaN when no second-best path exists. */
int assign_sisters(double *comp1, double *comp2, int n, int circular, int dim_mask,
                   int *state_path, double *total_cost, double *runner_up,
                   double *sister_a, double *sister_b) {
    if (n < (circular ? 3 : 2) || n > MAX_LOCI || dim_mask == 0 || (dim_mask & ~7)) return -1;
    for (int i = 0; i < n; i++) {
        double *a = comp1 + i*3, *b = comp2 + i*3;
        int swap = 0;
        for (int d = 0; d < 3; d++) { if (a[d] < b[d]) break; if (a[d] > b[d]) { swap = 1; break; } }
        if (swap) for (int d = 0; d < 3; d++) { double t = a[d]; a[d] = b[d]; b[d] = t; }
    }
    Hist hist[2][2]; int hn[2] = {1, 0};
    hist[0][0].cost = 0.0; hist[0][0].path[0] = 0; hist[0][0].len = 1;
    for (int i = 1; i < n; i++) {
        Hist nh[2][2]; int nn[2] = {0, 0};
        for (int s = 0; s < 2; s++)
            for (int ps = 0; ps < 2; ps++)
                for (int h = 0; h < hn[ps]; h++) {
                    Hist c = hist[ps][h];
                    c.cost += transition(comp1, comp2, n, i-1, ps, i, s, dim_mask);
                    c.path[c.len++] = (unsigned char)s;
                    best_two_insert(nh[s], &nn[s], &c);
                }
        memcpy(hist, nh, sizeof(hist)); hn[0] = nn[0]; hn[1] = nn[1];
    }
    Hist fin[2]; int fn = 0;
    for (int s = 0; s < 2; s++)
        for (int h = 0; h < hn[s]; h++) {
            Hist c = hist[s][h];
            if (circular) c.cost += transition(comp1, comp2, n, n-1, s, 0, 0, dim_mask);
            best_two_insert(fin, &fn, &c);
        }
    if (fn == 0) return -1;
    *total_cost = fin[0].cost;
    *runner_up = (fn > 1) ? fin[1].cost : NAN;
    for (int i = 0; i < n; i++) {
        state_path[i] = fin[0].path[i];
        const double *a = state_path[i] ? comp2 + i*3 : comp1 + i*3;
        const double *b = state_path[i] ? comp1 + i*3 : comp2 + i*3;
        memcpy(sister_a + i*3, a, 3*sizeof(double));
        memcpy(sister_b + i*3, b, 3*sizeof(double));
    }
    return 0;
}

/* ========================================================================
 * 8. Theta pattern classification on a circular chromosome (indices).
 *    Mirrors topology_core.analyze_theta_pattern.
 *    pattern_status: 0 not_theta_ori_not_doubled, 1 not_theta_one_sided,
 *                    2 fully_doubled_not_theta, 3 theta_pattern_supported,
 *                    4 theta_pattern_discordant
 *    trace_status:   0 theta_traceable, 1 theta_geometry_incomplete,
 *                    2 theta_not_traceable
 *    Returns 0, or -1 if doubled is not a subset of evaluable / bad input.
 * ======================================================================== */
int theta_pattern(int n, int ori, int ter, unsigned doubled, unsigned evaluable,
                  int *pattern_status, int *trace_status,
                  int *replicated, int *n_replicated,
                  int *unreplicated, int *n_unreplicated,
                  int *n_discordant, int *n_unevaluable) {
    if (n < 3 || n > 32 || ori < 0 || ori >= n || ter < 0 || ter >= n || ori == ter) return -1;
    unsigned universe = (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
    if ((doubled & ~universe) || (evaluable & ~universe) || (doubled & ~evaluable)) return -1;

    int armf[32], armr[32], nf = 0, nr = 0;
    for (int i = ori;; i = (i + 1) % n) { armf[nf++] = i; if (i == ter) break; }
    for (int i = ori;; i = (i - 1 + n) % n) { armr[nr++] = i; if (i == ter) break; }
    int runf = 0, runr = 0;
    while (runf < nf && (doubled >> armf[runf]) & 1u) runf++;
    while (runr < nr && (doubled >> armr[runr]) & 1u) runr++;
    unsigned coherent = 0;
    for (int i = 0; i < runf; i++) coherent |= 1u << armf[i];
    for (int i = 0; i < runr; i++) coherent |= 1u << armr[i];
    unsigned discordant = doubled & ~coherent;

    *n_replicated = 0; *n_unreplicated = 0;
    if (!((doubled >> ori) & 1u)) *pattern_status = 0;
    else if (runf < 2 || runr < 2) *pattern_status = 1;
    else if (coherent == universe) *pattern_status = 2;
    else {
        for (int i = runf - 1; i >= 0; i--) replicated[(*n_replicated)++] = armf[i];
        for (int i = 1; i < runr; i++) replicated[(*n_replicated)++] = armr[i];
        int ffront = armf[runf - 1], rfront = armr[runr - 1];
        for (int i = (ffront + 1) % n; i != rfront; i = (i + 1) % n)
            unreplicated[(*n_unreplicated)++] = i;
        if (*n_unreplicated == 0) return -1;
        *pattern_status = discordant ? 4 : 3;
    }
    *n_discordant = __builtin_popcount(discordant);
    *n_unevaluable = 0;
    for (int i = 0; i < *n_unreplicated; i++)
        if (!((evaluable >> unreplicated[i]) & 1u)) (*n_unevaluable)++;
    if (*pattern_status == 3 && *n_unevaluable == 0) *trace_status = 0;
    else if (*pattern_status == 3 || *pattern_status == 4) *trace_status = 1;
    else *trace_status = 2;
    return 0;
}

/* ========================================================================
 * 9. Geometry validation: exact segment distance, theta strands, loops.
 *    Mirrors topology_yamada._segment_distance and validators.
 * ======================================================================== */

static double segment_distance(const double *p0, const double *p1,
                               const double *q0, const double *q1) {
    double u[3], v[3], w[3];
    for (int d = 0; d < 3; d++) { u[d] = p1[d]-p0[d]; v[d] = q1[d]-q0[d]; w[d] = p0[d]-q0[d]; }
    double a = u[0]*u[0]+u[1]*u[1]+u[2]*u[2], b = u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
    double c = v[0]*v[0]+v[1]*v[1]+v[2]*v[2], d = u[0]*w[0]+u[1]*w[1]+u[2]*w[2];
    double e = v[0]*w[0]+v[1]*w[1]+v[2]*w[2];
    double den = a*c - b*b;
    double small = DBL_EPSILON * MAX(a*c, 1.0) * 32.0;
    double sn, sd = den, tn, td = den;
    if (den <= small) { sn = 0.0; sd = 1.0; tn = e; td = c; }
    else {
        sn = b*e - c*d; tn = a*e - b*d;
        if (sn < 0.0) { sn = 0.0; tn = e; td = c; }
        else if (sn > sd) { sn = sd; tn = e + b; td = c; }
    }
    if (tn < 0.0) { tn = 0.0; sn = MIN(MAX(-d, 0.0), a); sd = a; }
    else if (tn > td) { tn = td; sn = MIN(MAX(b - d, 0.0), a); sd = a; }
    double s = (fabs(sn) <= small) ? 0.0 : sn / sd;
    double t = (fabs(tn) <= small) ? 0.0 : tn / td;
    double r[3]; for (int k = 0; k < 3; k++) r[k] = w[k] + s*u[k] - t*v[k];
    return sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
}

static int overlap_from_shared_endpoint(const double *common, const double *ea,
                                        const double *eb, double tol) {
    double f[3], s[3];
    for (int d = 0; d < 3; d++) { f[d] = ea[d]-common[d]; s[d] = eb[d]-common[d]; }
    double fl = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]), sl = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    double cx = f[1]*s[2]-f[2]*s[1], cy = f[2]*s[0]-f[0]*s[2], cz = f[0]*s[1]-f[1]*s[0];
    double cross = sqrt(cx*cx+cy*cy+cz*cz);
    int collinear = cross <= tol * MAX(MAX(fl, sl), 1.0);
    int same_ray = (f[0]*s[0]+f[1]*s[1]+f[2]*s[2]) > 0.0;
    return collinear && same_ray;
}

static inline int pt_equal(const double *a, const double *b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

/* Return codes: 0 ok; 1 need 3 strands; 2 strand too short; 3 forks coincide;
 * 4 strand endpoints not shared exactly; 5 zero-length segment;
 * 6 self-intersection; 7 overlap beyond fork 1; 8 overlap beyond fork 2;
 * 9 inter-strand contact. *which reports the offending strand index. */
int validate_theta_strands(const double *pts, const int *lens, int n_strands,
                           double tol, int *which) {
    *which = -1;
    if (n_strands != 3) return 1;
    const double *S[3]; int L[3], off = 0;
    for (int i = 0; i < 3; i++) { S[i] = pts + off*3; L[i] = lens[i]; off += lens[i];
        if (L[i] < 2) { *which = i; return 2; } }
    const double *start = S[0], *end = S[0] + (L[0]-1)*3;
    if (pt_equal(start, end)) return 3;
    for (int i = 1; i < 3; i++)
        if (!pt_equal(S[i], start) || !pt_equal(S[i] + (L[i]-1)*3, end)) { *which = i; return 4; }
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < L[i]-1; k++) if (pt_equal(S[i]+k*3, S[i]+(k+1)*3)) { *which = i; return 5; }
        for (int a = 0; a < L[i]-1; a++)
            for (int b = a+2; b < L[i]-1; b++)
                if (segment_distance(S[i]+a*3, S[i]+(a+1)*3, S[i]+b*3, S[i]+(b+1)*3) <= tol) { *which = i; return 6; }
    }
    for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++) {
        const double *A = S[i], *B = S[j]; int la = L[i], lb = L[j];
        for (int a = 0; a < la-1; a++) for (int b = 0; b < lb-1; b++) {
            if (a == 0 && b == 0) {
                if (overlap_from_shared_endpoint(A, A+3, B+3, tol)) return 7;
                continue;
            }
            if (a == la-2 && b == lb-2) {
                if (overlap_from_shared_endpoint(A+(la-1)*3, A+(la-2)*3, B+(lb-2)*3, tol)) return 8;
                continue;
            }
            if (segment_distance(A+a*3, A+(a+1)*3, B+b*3, B+(b+1)*3) <= tol) return 9;
        }
    }
    return 0;
}

/* Prepare a closed loop: drops an exact repeated closing point, validates
 * simplicity, writes closed vertices (n+1 points) to out. Returns the count
 * of closed points, or a negative error: -1 too few points, -2 zero segment,
 * -3 self-intersection. */
static int prepare_loop(const double *pts, int n, double tol, double *out) {
    if (n < 3) return -1;
    if (pt_equal(pts, pts + (n-1)*3)) n--;
    if (n < 3) return -1;
    memcpy(out, pts, (size_t)n*3*sizeof(double));
    memcpy(out + n*3, pts, 3*sizeof(double));
    for (int k = 0; k < n; k++) {
        double *p = out + k*3, *q = out + (k+1)*3;
        double dx = q[0]-p[0], dy = q[1]-p[1], dz = q[2]-p[2];
        if (sqrt(dx*dx+dy*dy+dz*dz) <= tol) return -2;
    }
    for (int a = 0; a < n; a++) for (int b = a+1; b < n; b++) {
        if (b == a+1 || (a == 0 && b == n-1)) continue;
        if (segment_distance(out+a*3, out+(a+1)*3, out+b*3, out+(b+1)*3) <= tol) return -3;
    }
    return n + 1;
}

static double min_interloop_distance(const double *A, int na, const double *B, int nb) {
    double best = DBL_MAX;
    for (int i = 0; i < na-1; i++) for (int j = 0; j < nb-1; j++) {
        double d = segment_distance(A+i*3, A+(i+1)*3, B+j*3, B+(j+1)*3);
        if (d < best) best = d;
    }
    return best;
}

/* ========================================================================
 * 10. Classical linking number via deterministic projection sweep.
 *     Mirrors topology_yamada.compute_linking_number. The Fibonacci-sphere
 *     phase is supplied by the caller (drawn with numpy for reproducibility).
 * ======================================================================== */

static void projection_basis(const double *dir, double *first, double *second) {
    int k = 0;
    if (fabs(dir[1]) < fabs(dir[k])) k = 1;
    if (fabs(dir[2]) < fabs(dir[k])) k = 2;
    double axis[3] = {0,0,0}; axis[k] = 1.0;
    double ad = axis[0]*dir[0]+axis[1]*dir[1]+axis[2]*dir[2];
    for (int d = 0; d < 3; d++) first[d] = axis[d] - dir[d]*ad;
    double nrm = sqrt(first[0]*first[0]+first[1]*first[1]+first[2]*first[2]);
    for (int d = 0; d < 3; d++) first[d] /= nrm;
    second[0] = dir[1]*first[2]-dir[2]*first[1];
    second[1] = dir[2]*first[0]-dir[0]*first[2];
    second[2] = dir[0]*first[1]-dir[1]*first[0];
}

#define LINK_NONGENERIC 0x7fffffff

static int linking_in_projection(const double *A, int na, const double *B, int nb,
                                 const double *dir, double span) {
    const double rtol = 1e-10;
    double f[3], s[3]; projection_basis(dir, f, s);
    int signed_sum = 0;
    for (int i = 0; i < na-1; i++) {
        const double *a0 = A+i*3, *a1 = A+(i+1)*3;
        double pa[2] = {a0[0]*f[0]+a0[1]*f[1]+a0[2]*f[2], a0[0]*s[0]+a0[1]*s[1]+a0[2]*s[2]};
        double qa[2] = {a1[0]*f[0]+a1[1]*f[1]+a1[2]*f[2], a1[0]*s[0]+a1[1]*s[1]+a1[2]*s[2]};
        double ea[2] = {qa[0]-pa[0], qa[1]-pa[1]};
        double ta[3] = {a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]};
        double da0 = a0[0]*dir[0]+a0[1]*dir[1]+a0[2]*dir[2], da1 = a1[0]*dir[0]+a1[1]*dir[1]+a1[2]*dir[2];
        double nea = sqrt(ea[0]*ea[0]+ea[1]*ea[1]);
        for (int j = 0; j < nb-1; j++) {
            const double *b0 = B+j*3, *b1 = B+(j+1)*3;
            double pb[2] = {b0[0]*f[0]+b0[1]*f[1]+b0[2]*f[2], b0[0]*s[0]+b0[1]*s[1]+b0[2]*s[2]};
            double qb[2] = {b1[0]*f[0]+b1[1]*f[1]+b1[2]*f[2], b1[0]*s[0]+b1[1]*s[1]+b1[2]*s[2]};
            double eb[2] = {qb[0]-pb[0], qb[1]-pb[1]};
            double tb[3] = {b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2]};
            double den = ea[0]*eb[1] - ea[1]*eb[0];
            double scale = MAX(nea * sqrt(eb[0]*eb[0]+eb[1]*eb[1]), 1.0);
            double dl[2] = {pb[0]-pa[0], pb[1]-pa[1]};
            if (fabs(den) <= rtol * scale) {
                if (fabs(dl[0]*ea[1] - dl[1]*ea[0]) <= rtol * scale) return LINK_NONGENERIC;
                continue;
            }
            double along_a = (dl[0]*eb[1] - dl[1]*eb[0]) / den;
            double along_b = (dl[0]*ea[1] - dl[1]*ea[0]) / den;
            double et = rtol * 100.0;
            int near_edge = (-et <= along_a && along_a <= et) || (1.0-et <= along_a && along_a <= 1.0+et) ||
                            (-et <= along_b && along_b <= et) || (1.0-et <= along_b && along_b <= 1.0+et);
            if (near_edge) {
                if (-et <= along_a && along_a <= 1.0+et && -et <= along_b && along_b <= 1.0+et)
                    return LINK_NONGENERIC;
                continue;
            }
            if (!(0.0 < along_a && along_a < 1.0 && 0.0 < along_b && along_b < 1.0)) continue;
            double db0 = b0[0]*dir[0]+b0[1]*dir[1]+b0[2]*dir[2], db1 = b1[0]*dir[0]+b1[1]*dir[1]+b1[2]*dir[2];
            double da = da0 + along_a*(da1-da0), db = db0 + along_b*(db1-db0);
            if (fabs(da - db) <= rtol * span) return LINK_NONGENERIC;
            double cx, cy, cz;
            if (da > db) { cx = ta[1]*tb[2]-ta[2]*tb[1]; cy = ta[2]*tb[0]-ta[0]*tb[2]; cz = ta[0]*tb[1]-ta[1]*tb[0]; }
            else         { cx = tb[1]*ta[2]-tb[2]*ta[1]; cy = tb[2]*ta[0]-tb[0]*ta[2]; cz = tb[0]*ta[1]-tb[1]*ta[0]; }
            double orient = cx*dir[0]+cy*dir[1]+cz*dir[2];
            if (fabs(orient) <= rtol * span * span) return LINK_NONGENERIC;
            signed_sum += orient > 0 ? 1 : -1;
        }
    }
    if (signed_sum % 2) return LINK_NONGENERIC;
    return signed_sum / 2;
}

/* loop_a/loop_b: raw vertices [na][3], [nb][3] (closing point optional).
 * values_out: [n_projections] linking values, LINK_NONGENERIC where skipped.
 * Returns 0 ok; -1/-2/-3 loop A prep error (see prepare_loop); -11/-12/-13 loop B;
 * -20 loops not disjoint; -21 no generic projection; -30 input too large. */
int compute_linking(const double *loop_a, int na, const double *loop_b, int nb,
                    int n_projections, double phase, double tol,
                    int *values_out, int *linking_number, int *n_valid, int *modal_count,
                    double *min_distance) {
    double A[(MAX_POINTS+1)*3], B[(MAX_POINTS+1)*3];
    if (na > MAX_POINTS || nb > MAX_POINTS) return -30;
    int ca = prepare_loop(loop_a, na, tol, A); if (ca < 0) return ca;
    int cb = prepare_loop(loop_b, nb, tol, B); if (cb < 0) return cb - 10;
    *min_distance = min_interloop_distance(A, ca, B, cb);
    if (*min_distance <= tol) return -20;

    double lo[3] = {DBL_MAX,DBL_MAX,DBL_MAX}, hi[3] = {-DBL_MAX,-DBL_MAX,-DBL_MAX};
    for (int i = 0; i < ca; i++) for (int d = 0; d < 3; d++) { lo[d] = MIN(lo[d], A[i*3+d]); hi[d] = MAX(hi[d], A[i*3+d]); }
    for (int i = 0; i < cb; i++) for (int d = 0; d < 3; d++) { lo[d] = MIN(lo[d], B[i*3+d]); hi[d] = MAX(hi[d], B[i*3+d]); }
    double span = MAX(MAX(hi[0]-lo[0], hi[1]-lo[1]), MAX(hi[2]-lo[2], 1.0));

    const double golden = M_PI * (3.0 - sqrt(5.0));
    *n_valid = 0;
    for (int i = 0; i < n_projections; i++) {
        double z = 1.0 - 2.0 * (i + 0.5) / n_projections;
        double r = sqrt(MAX(0.0, 1.0 - z*z));
        double ang = phase + i * golden;
        double dir[3] = {r*cos(ang), r*sin(ang), z};
        int v = linking_in_projection(A, ca, B, cb, dir, span);
        values_out[i] = v;
        if (v != LINK_NONGENERIC) (*n_valid)++;
    }
    if (*n_valid == 0) return -21;
    /* modal value; tie-break by (-count, |value|, value) */
    int best_v = 0, best_c = -1;
    for (int i = 0; i < n_projections; i++) {
        int v = values_out[i]; if (v == LINK_NONGENERIC) continue;
        int c = 0;
        for (int j = 0; j < n_projections; j++) if (values_out[j] == v) c++;
        if (c > best_c || (c == best_c && (abs(v) < abs(best_v) || (abs(v) == abs(best_v) && v < best_v)))) {
            best_c = c; best_v = v;
        }
    }
    *linking_number = best_v; *modal_count = best_c;
    return 0;
}

/* ========================================================================
 * 11. Perturbation stability of the linking number.
 *     Both loops are re-drawn `draws` times with independent Gaussian noise
 *     (sigma_xy on x and y, sigma_z on z, xorshift64* + Box-Muller, seeded)
 *     and the modal linking number of each draw is compared to `nominal`.
 *     Draws whose perturbed loops fail validation or have no generic
 *     projection are counted in *n_invalid and do not count as agreement.
 * ======================================================================== */

static inline double unit_uniform(unsigned long long *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    unsigned long long x = *s * 0x2545F4914F6CDD1DULL;
    return ((double)(x >> 11) + 0.5) / (double)(1ULL << 53);
}

static inline double gaussian(unsigned long long *s) {
    double u1 = unit_uniform(s), u2 = unit_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int link_perturbation(const double *loop_a, int na, const double *loop_b, int nb,
                      int n_projections, double phase, double tol,
                      int draws, double sigma_xy, double sigma_z,
                      unsigned long long seed, int nominal,
                      int *n_agree, int *n_invalid) {
    if (na > MAX_POINTS || nb > MAX_POINTS || na < 1 || nb < 1) return -30;
    if (draws < 1 || n_projections < 1) return -31;
    double A[MAX_POINTS*3], B[MAX_POINTS*3];
    int values[n_projections];
    unsigned long long rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    *n_agree = 0; *n_invalid = 0;
    for (int d = 0; d < draws; d++) {
        for (int i = 0; i < na; i++) {
            A[i*3]   = loop_a[i*3]   + sigma_xy * gaussian(&rng);
            A[i*3+1] = loop_a[i*3+1] + sigma_xy * gaussian(&rng);
            A[i*3+2] = loop_a[i*3+2] + sigma_z  * gaussian(&rng);
        }
        for (int i = 0; i < nb; i++) {
            B[i*3]   = loop_b[i*3]   + sigma_xy * gaussian(&rng);
            B[i*3+1] = loop_b[i*3+1] + sigma_xy * gaussian(&rng);
            B[i*3+2] = loop_b[i*3+2] + sigma_z  * gaussian(&rng);
        }
        int ln, nv, modal; double dist;
        int rc = compute_linking(A, na, B, nb, n_projections, phase, tol,
                                 values, &ln, &nv, &modal, &dist);
        if (rc != 0) { (*n_invalid)++; continue; }
        if (ln == nominal) (*n_agree)++;
    }
    return 0;
}

/* ========================================================================
 * 12. Perturbation stability of a theta graph's Yamada polynomial.
 *     Observed loci (single-copy path, sister A, sister B) are re-drawn with
 *     Gaussian noise, the two fork vertices are recomputed from the perturbed
 *     neighbours (boundary-midpoint or pair-midpoint model), the strands are
 *     validated and the normalized polynomial recomputed and compared with
 *     `nominal_norm` (POLY_SIZE coefficients). Draws failing validation or
 *     without a generic projection count in *n_invalid; draws above the
 *     crossing cap count in *n_unknown; neither counts as agreement.
 * ======================================================================== */

static void midpoint3(const double *a, const double *b, double *out) {
    for (int d = 0; d < 3; d++) out[d] = (a[d] + b[d]) / 2.0;
}

int theta_perturbation(const double *singles, int n_single,
                       const double *sister_a, const double *sister_b, int n_double,
                       int boundary_midpoint, double tol, int tries,
                       unsigned long long seed, int max_crossings,
                       int draws, double sigma_xy, double sigma_z,
                       const int *nominal_norm,
                       int *n_agree, int *n_invalid, int *n_unknown) {
    if (n_single < 1 || n_double < 1 || n_single + 2 * n_double + 6 > MAX_POINTS) return -30;
    if (draws < 1) return -31;
    double S[MAX_POINTS*3], A[MAX_POINTS*3], B[MAX_POINTS*3];
    double pts[MAX_POINTS*3];
    int lens[3] = {n_single + 2, n_double + 2, n_double + 2};
    unsigned long long rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    *n_agree = 0; *n_invalid = 0; *n_unknown = 0;
    for (int d = 0; d < draws; d++) {
        for (int i = 0; i < n_single; i++) {
            S[i*3]   = singles[i*3]   + sigma_xy * gaussian(&rng);
            S[i*3+1] = singles[i*3+1] + sigma_xy * gaussian(&rng);
            S[i*3+2] = singles[i*3+2] + sigma_z  * gaussian(&rng);
        }
        for (int i = 0; i < n_double; i++) {
            A[i*3]   = sister_a[i*3]   + sigma_xy * gaussian(&rng);
            A[i*3+1] = sister_a[i*3+1] + sigma_xy * gaussian(&rng);
            A[i*3+2] = sister_a[i*3+2] + sigma_z  * gaussian(&rng);
            B[i*3]   = sister_b[i*3]   + sigma_xy * gaussian(&rng);
            B[i*3+1] = sister_b[i*3+1] + sigma_xy * gaussian(&rng);
            B[i*3+2] = sister_b[i*3+2] + sigma_z  * gaussian(&rng);
        }
        double f1[3], f2[3], m1[3], m2[3];
        midpoint3(A, B, m1);
        midpoint3(A + (n_double-1)*3, B + (n_double-1)*3, m2);
        if (boundary_midpoint) { midpoint3(m1, S, f1); midpoint3(m2, S + (n_single-1)*3, f2); }
        else { memcpy(f1, m1, sizeof f1); memcpy(f2, m2, sizeof f2); }
        int k = 0;
        memcpy(pts + (k++)*3, f1, 3*sizeof(double));
        for (int i = 0; i < n_single; i++) memcpy(pts + (k++)*3, S + i*3, 3*sizeof(double));
        memcpy(pts + (k++)*3, f2, 3*sizeof(double));
        memcpy(pts + (k++)*3, f1, 3*sizeof(double));
        for (int i = 0; i < n_double; i++) memcpy(pts + (k++)*3, A + i*3, 3*sizeof(double));
        memcpy(pts + (k++)*3, f2, 3*sizeof(double));
        memcpy(pts + (k++)*3, f1, 3*sizeof(double));
        for (int i = 0; i < n_double; i++) memcpy(pts + (k++)*3, B + i*3, 3*sizeof(double));
        memcpy(pts + (k++)*3, f2, 3*sizeof(double));
        int which;
        if (validate_theta_strands(pts, lens, 3, tol, &which) != 0) { (*n_invalid)++; continue; }
        LPoly poly, norm;
        int ret = yamada_polynomial(pts, lens, 3, tol, tries, seed + 1 + (unsigned long long)d,
                                    max_crossings, &poly);
        if (ret == -2) { (*n_unknown)++; continue; }
        if (ret < 0) { (*n_invalid)++; continue; }
        poly_normalize(&poly, &norm);
        int same = 1;
        for (int e = -MAX_EXP; e <= MAX_EXP && same; e++) {
            int c = (e >= norm.lo && e <= norm.hi) ? norm.c[e + OFFSET] : 0;
            if (c != nominal_norm[e + OFFSET]) same = 0;
        }
        if (same) (*n_agree)++;
    }
    return 0;
}

/* ---- ctypes entry points ---- */

/* Returns crossings (>=0) or a negative code (-1 no generic projection,
 * -2 exceeds max_crossings, -3 input too large). Fills raw and normalized
 * coefficient arrays of length POLY_SIZE and the graph vertex count. */
int compute_yamada(const double *strand_points, const int *strand_lens, int n_strands,
                   double tol, int tries, unsigned long long seed, int max_crossings,
                   int *out_raw_coeff, int *out_norm_coeff, int *out_n_crossings,
                   int *out_n_vertices) {
    LPoly poly, norm;
    g_overflow = 0;
    memset(out_raw_coeff, 0, POLY_SIZE * sizeof(int));
    memset(out_norm_coeff, 0, POLY_SIZE * sizeof(int));
    *out_n_crossings = 0; *out_n_vertices = 0;
    int ret = yamada_polynomial(strand_points, strand_lens, n_strands, tol, tries, seed,
                                max_crossings, &poly);
    if (ret < 0) return ret;
    /* vertex count: distinct strand endpoints within tol */
    {
        int nv = 0, total = 0; double verts[MAX_STRANDS*2*3];
        for (int si = 0; si < n_strands; si++) {
            for (int ep = 0; ep < 2; ep++) {
                const double *p = strand_points + (total + (ep ? strand_lens[si]-1 : 0))*3;
                int found = 0;
                for (int vi = 0; vi < nv; vi++) {
                    double dx=p[0]-verts[vi*3], dy=p[1]-verts[vi*3+1], dz=p[2]-verts[vi*3+2];
                    if (dx*dx+dy*dy+dz*dz < tol*tol) { found = 1; break; }
                }
                if (!found) { memcpy(verts + nv*3, p, 3*sizeof(double)); nv++; }
            }
            total += strand_lens[si];
        }
        *out_n_vertices = nv;
    }
    *out_n_crossings = ret;
    for (int e = poly.lo; e <= poly.hi; e++) out_raw_coeff[e + OFFSET] = poly.c[e + OFFSET];
    poly_normalize(&poly, &norm);
    for (int e = norm.lo; e <= norm.hi; e++) out_norm_coeff[e + OFFSET] = norm.c[e + OFFSET];
    return ret;
}

int get_link_nongeneric_sentinel(void) { return LINK_NONGENERIC; }
int get_max_loci(void) { return MAX_LOCI; }

int  get_poly_size(void)      { return POLY_SIZE; }
int  get_offset(void)         { return OFFSET; }
int  get_max_crossings(void)  { return MAX_CROSSINGS; }
int  get_overflow_flag(void)  { return g_overflow; }
long long get_last_states(void) { return g_last_states; }
long long get_last_hits(void)   { return g_last_hits; }
long long get_last_misses(void) { return g_last_misses; }
long long get_last_resets(void) { return g_last_resets; }
void set_memo_budget_mb(int mb) { if (mb > 0) g_memo_budget = (size_t)mb << 20; }
int get_memo_budget_mb(void) { return (int)(g_memo_budget >> 20); }
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
