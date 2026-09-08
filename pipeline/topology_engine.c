/*
 * topology_engine.c — C implementation of the Yamada polynomial engine
 * and topology detection for theta-curves and two-loop links.
 *
 * Replaces: yamada.py, topology_core.py, topology_yamada.py
 * Called from: detect_topology.py (Python orchestrator) via ctypes
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

/* ========================================================================
 * 1. Laurent polynomial in variable A
 *    Stored as coefficient array: coeff[i] is the coefficient of A^(i-OFFSET).
 *    Exponent range: [-MAX_EXP, +MAX_EXP].
 * ======================================================================== */

#define MAX_EXP 64
#define POLY_SIZE (2 * MAX_EXP + 1)
#define OFFSET MAX_EXP

typedef struct {
    int coeff[POLY_SIZE];
} LPoly;

static const LPoly POLY_ZERO = {{0}};

static LPoly poly_one(void) {
    LPoly p = POLY_ZERO;
    p.coeff[OFFSET] = 1;
    return p;
}

static LPoly poly_neg_one(void) {
    LPoly p = POLY_ZERO;
    p.coeff[OFFSET] = -1;
    return p;
}

/* sigma = A + 1 + A^{-1} */
static LPoly poly_neg_sigma(void) {
    LPoly p = POLY_ZERO;
    p.coeff[OFFSET + 1] = -1;
    p.coeff[OFFSET]     = -1;
    p.coeff[OFFSET - 1] = -1;
    return p;
}

static LPoly poly_add(const LPoly *a, const LPoly *b) {
    LPoly r;
    for (int i = 0; i < POLY_SIZE; i++)
        r.coeff[i] = a->coeff[i] + b->coeff[i];
    return r;
}

static LPoly poly_mul(const LPoly *a, const LPoly *b) {
    LPoly r = POLY_ZERO;
    for (int i = 0; i < POLY_SIZE; i++) {
        if (a->coeff[i] == 0) continue;
        for (int j = 0; j < POLY_SIZE; j++) {
            if (b->coeff[j] == 0) continue;
            int exp = (i - OFFSET) + (j - OFFSET);
            if (exp >= -MAX_EXP && exp <= MAX_EXP)
                r.coeff[exp + OFFSET] += a->coeff[i] * b->coeff[j];
        }
    }
    return r;
}

static LPoly poly_scale_mono(const LPoly *p, int exp, int coef) {
    LPoly r = POLY_ZERO;
    for (int i = 0; i < POLY_SIZE; i++) {
        if (p->coeff[i] == 0) continue;
        int new_exp = (i - OFFSET) + exp;
        if (new_exp >= -MAX_EXP && new_exp <= MAX_EXP)
            r.coeff[new_exp + OFFSET] += p->coeff[i] * coef;
    }
    return r;
}

static int poly_is_zero(const LPoly *p) {
    for (int i = 0; i < POLY_SIZE; i++)
        if (p->coeff[i] != 0) return 0;
    return 1;
}

static int poly_min_exp(const LPoly *p) {
    for (int i = 0; i < POLY_SIZE; i++)
        if (p->coeff[i] != 0) return i - OFFSET;
    return 0;
}

/* Normalize: shift min exponent to 0, canonical sign via (-A)^n rule. */
static LPoly poly_normalize(const LPoly *p) {
    if (poly_is_zero(p)) return POLY_ZERO;
    int m = poly_min_exp(p);
    int sign = ((-m) % 2 != 0) ? -1 : 1;
    LPoly r = POLY_ZERO;
    for (int i = 0; i < POLY_SIZE; i++) {
        if (p->coeff[i] == 0) continue;
        int new_i = (i - OFFSET) - m + OFFSET;
        if (new_i >= 0 && new_i < POLY_SIZE)
            r.coeff[new_i] = sign * p->coeff[i];
    }
    return r;
}

static int poly_equal(const LPoly *a, const LPoly *b) {
    return memcmp(a->coeff, b->coeff, sizeof(a->coeff)) == 0;
}

/* Compare up to multiplication by (-A)^n. */
static int poly_equal_up_to_unit(const LPoly *a, const LPoly *b) {
    LPoly na = poly_normalize(a);
    LPoly nb = poly_normalize(b);
    return poly_equal(&na, &nb);
}

/* Write polynomial to a JSON-like string: {exp: coeff, ...}
 * buf must be large enough. Returns length written. */
static int poly_to_json(const LPoly *p, char *buf, int bufsize) {
    int pos = 0;
    pos += snprintf(buf + pos, bufsize - pos, "{");
    int first = 1;
    for (int i = 0; i < POLY_SIZE; i++) {
        if (p->coeff[i] == 0) continue;
        if (!first) pos += snprintf(buf + pos, bufsize - pos, ",");
        pos += snprintf(buf + pos, bufsize - pos, "\"%d\":%d", i - OFFSET, p->coeff[i]);
        first = 0;
    }
    pos += snprintf(buf + pos, bufsize - pos, "}");
    return pos;
}

/* ========================================================================
 * 2. Graph R — Yamada graph invariant via deletion-contraction
 *    Memoized with a hash table.
 * ======================================================================== */

#define MAX_GRAPH_EDGES 128
#define MEMO_BUCKETS 65521   /* prime */

typedef struct {
    int u, v;
} Edge;

typedef struct {
    int iso;            /* isolated vertex count */
    int n_edges;
    Edge edges[MAX_GRAPH_EDGES];
} CanonGraph;

typedef struct MemoEntry {
    CanonGraph key;
    LPoly val;
    struct MemoEntry *next;
} MemoEntry;

static MemoEntry *memo_table[MEMO_BUCKETS];
static int memo_count = 0;

static unsigned int graph_hash(const CanonGraph *g) {
    unsigned int h = (unsigned int)g->iso * 2654435761u;
    for (int i = 0; i < g->n_edges; i++) {
        h ^= (unsigned int)(g->edges[i].u * 31 + g->edges[i].v) * 2246822519u;
        h = (h << 13) | (h >> 19);
    }
    return h % MEMO_BUCKETS;
}

static int graph_equal(const CanonGraph *a, const CanonGraph *b) {
    if (a->iso != b->iso || a->n_edges != b->n_edges) return 0;
    return memcmp(a->edges, b->edges, (size_t)a->n_edges * sizeof(Edge)) == 0;
}

static LPoly *memo_get(const CanonGraph *key) {
    unsigned int h = graph_hash(key);
    for (MemoEntry *e = memo_table[h]; e; e = e->next)
        if (graph_equal(&e->key, key)) return &e->val;
    return NULL;
}

static void memo_put(const CanonGraph *key, const LPoly *val) {
    unsigned int h = graph_hash(key);
    MemoEntry *e = (MemoEntry *)malloc(sizeof(MemoEntry));
    e->key = *key;
    e->val = *val;
    e->next = memo_table[h];
    memo_table[h] = e;
    memo_count++;
}

void memo_clear(void) {
    for (int i = 0; i < MEMO_BUCKETS; i++) {
        MemoEntry *e = memo_table[i];
        while (e) { MemoEntry *next = e->next; free(e); e = next; }
        memo_table[i] = NULL;
    }
    memo_count = 0;
}

static int edge_cmp(const void *a, const void *b) {
    const Edge *ea = (const Edge *)a, *eb = (const Edge *)b;
    if (ea->u != eb->u) return ea->u - eb->u;
    return ea->v - eb->v;
}

static CanonGraph canonicalize(int num_v, const Edge *edges, int n_edges) {
    /* Relabel vertices by first appearance in sorted edge list. */
    Edge sorted[MAX_GRAPH_EDGES];
    for (int i = 0; i < n_edges; i++) {
        sorted[i].u = edges[i].u < edges[i].v ? edges[i].u : edges[i].v;
        sorted[i].v = edges[i].u < edges[i].v ? edges[i].v : edges[i].u;
    }
    qsort(sorted, (size_t)n_edges, sizeof(Edge), edge_cmp);

    int map[512];
    memset(map, -1, sizeof(map));
    int next_id = 0;
    CanonGraph g;
    g.n_edges = n_edges;
    for (int i = 0; i < n_edges; i++) {
        if (map[sorted[i].u] < 0) map[sorted[i].u] = next_id++;
        if (map[sorted[i].v] < 0) map[sorted[i].v] = next_id++;
        int a = map[sorted[i].u], b = map[sorted[i].v];
        g.edges[i].u = a < b ? a : b;
        g.edges[i].v = a < b ? b : a;
    }
    qsort(g.edges, (size_t)n_edges, sizeof(Edge), edge_cmp);
    g.iso = num_v - next_id;
    return g;
}

static int count_vertices(const Edge *edges, int n) {
    int seen[512]; memset(seen, 0, sizeof(seen));
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (!seen[edges[i].u]) { seen[edges[i].u] = 1; cnt++; }
        if (!seen[edges[i].v]) { seen[edges[i].v] = 1; cnt++; }
    }
    return cnt;
}

static LPoly graph_R(int num_v, const Edge *edges, int n_edges);

static LPoly graph_R_canon(int iso, const Edge *edges, int n_edges) {
    LPoly base = (iso % 2 == 0) ? poly_one() : poly_neg_one();

    if (n_edges == 0) return base;

    /* Strip loops */
    int n_loops = 0;
    Edge non_loop[MAX_GRAPH_EDGES];
    int n_nl = 0;
    for (int i = 0; i < n_edges; i++) {
        if (edges[i].u == edges[i].v) n_loops++;
        else non_loop[n_nl++] = edges[i];
    }
    if (n_loops > 0) {
        LPoly sub = graph_R(count_vertices(edges, n_edges), non_loop, n_nl);
        LPoly ns = poly_neg_sigma();
        LPoly f = poly_one();
        for (int i = 0; i < n_loops; i++)
            f = poly_mul(&f, &ns);
        LPoly prod = poly_mul(&f, &sub);
        if (iso) return poly_mul(&base, &prod);
        return prod;
    }

    /* Deletion-contraction on first edge */
    int u = edges[0].u, v = edges[0].v;
    int nv = count_vertices(edges, n_edges);

    /* Delete */
    LPoly R_del = graph_R(nv, edges + 1, n_edges - 1);

    /* Contract: merge v into u */
    Edge merged[MAX_GRAPH_EDGES];
    int nm = 0;
    for (int i = 1; i < n_edges; i++) {
        int a = edges[i].u == v ? u : edges[i].u;
        int b = edges[i].v == v ? u : edges[i].v;
        merged[nm].u = a; merged[nm].v = b; nm++;
    }
    LPoly R_con = graph_R(nv - 1, merged, nm);

    LPoly val = poly_add(&R_del, &R_con);
    if (iso) return poly_mul(&base, &val);
    return val;
}

static LPoly graph_R(int num_v, const Edge *edges, int n_edges) {
    if (n_edges > MAX_GRAPH_EDGES) n_edges = MAX_GRAPH_EDGES;
    CanonGraph key = canonicalize(num_v, edges, n_edges);
    LPoly *cached = memo_get(&key);
    if (cached) return *cached;
    LPoly val = graph_R_canon(key.iso, key.edges, key.n_edges);
    memo_put(&key, &val);
    return val;
}

/* ========================================================================
 * 3. Crossing detection: 2D segment-segment intersection with depth
 * ======================================================================== */

#define MAX_CROSSINGS 4096

typedef struct {
    int seg_i, seg_j;
    double ti, tj;
    double zi, zj;
    double px, py;
} CrossingRecord;

/* Find all crossings between projected segments.
 * segments: [n_seg][6] = (x0,y0,z0,x1,y1,z1) after rotation.
 * meta: [n_seg][3] = (strand_id, endpoint_id_0, endpoint_id_1).
 * Returns number of crossings found; fills `out`.
 * Returns -1 if non-generic (degenerate) projection. */
static int find_crossings(const double *segments, const int *meta, int n_seg,
                           CrossingRecord *out) {
    int count = 0;
    for (int i = 0; i < n_seg && count < MAX_CROSSINGS; i++) {
        const double *a = segments + i * 6;
        for (int j = i + 1; j < n_seg && count < MAX_CROSSINGS; j++) {
            const double *b = segments + j * 6;
            /* Skip if segments share an endpoint */
            const int *mi = meta + i * 3;
            const int *mj = meta + j * 3;
            if (mi[1] == mj[1] || mi[1] == mj[2] ||
                mi[2] == mj[1] || mi[2] == mj[2]) continue;

            double dax = a[3] - a[0], day = a[4] - a[1];
            double dbx = b[3] - b[0], dby = b[4] - b[1];
            double denom = dax * dby - day * dbx;
            if (fabs(denom) < 1e-9) continue;

            double ox = b[0] - a[0], oy = b[1] - a[1];
            double ta = (ox * dby - oy * dbx) / denom;
            double tb = (ox * day - oy * dax) / denom;

            if (ta > 1e-7 && ta < 1.0 - 1e-7 && tb > 1e-7 && tb < 1.0 - 1e-7) {
                double za = a[2] * (1.0 - ta) + a[5] * ta;
                double zb = b[2] * (1.0 - tb) + b[5] * tb;
                if (fabs(za - zb) < 1e-9) return -1;

                out[count].seg_i = i;
                out[count].seg_j = j;
                out[count].ti = ta;
                out[count].tj = tb;
                out[count].zi = za;
                out[count].zj = zb;
                out[count].px = a[0] + dax * ta;
                out[count].py = a[1] + day * ta;
                count++;
            } else if (ta >= -1e-7 && ta <= 1.0 + 1e-7 &&
                       tb >= -1e-7 && tb <= 1.0 + 1e-7) {
                return -1; /* near-degenerate */
            }
        }
    }
    /* Check crossing-crossing separation */
    for (int i = 0; i < count; i++)
        for (int j = i + 1; j < count; j++) {
            double dx = out[i].px - out[j].px;
            double dy = out[i].py - out[j].py;
            if (dx * dx + dy * dy < 1e-12) return -1;
        }
    return count;
}

/* ========================================================================
 * 4. Diagram building + skein resolution
 * ======================================================================== */

/* A diagram edge endpoint: either a graph vertex or a crossing port. */
typedef struct {
    int is_vertex;   /* 1 = graph vertex, 0 = crossing */
    int id;          /* vertex index or crossing index */
    int port;        /* crossing port (0..3) if is_vertex==0, unused otherwise */
} DiagramEndpoint;

typedef struct {
    DiagramEndpoint a, b;
} DiagramEdge;

#define MAX_DIAGRAM_EDGES 512
#define MAX_STRANDS 16
#define MAX_SEGMENTS 256
#define MAX_POINTS 512

typedef struct {
    int n_vertices;
    double vertices[MAX_POINTS * 3];   /* [n_vertices][3] */
    int n_strands;
    int strand_start[MAX_STRANDS];     /* index into points array */
    int strand_len[MAX_STRANDS];       /* number of points per strand */
    double points[MAX_POINTS * 3];     /* all strand points concatenated */
    int strand_endv[MAX_STRANDS * 2];  /* [strand][0/1] = vertex id of start/end */
} SpatialGraph;

/* Build segments + metadata from spatial graph for crossing detection.
 * Point IDs use coincidence classes: points within tol of each other
 * share the same ID, so the crossing detector skips adjacent segments. */
static int build_segments(const SpatialGraph *sg, double tol,
                          double *seg6, int *meta3, int *seg_loc2) {
    /* Assign coincidence-class IDs to all points */
    int total_pts = 0;
    for (int si = 0; si < sg->n_strands; si++)
        total_pts += sg->strand_len[si];

    int pid[MAX_POINTS];
    for (int i = 0; i < total_pts; i++) pid[i] = i;
    for (int i = 0; i < total_pts; i++) {
        const double *pi = sg->points + i * 3;
        for (int j = i + 1; j < total_pts; j++) {
            double dx = pi[0] - sg->points[j*3];
            double dy = pi[1] - sg->points[j*3+1];
            double dz = pi[2] - sg->points[j*3+2];
            if (sqrt(dx*dx+dy*dy+dz*dz) < tol)
                pid[j] = pid[i];
        }
    }

    int n = 0;
    for (int si = 0; si < sg->n_strands; si++) {
        int start = sg->strand_start[si];
        int len = sg->strand_len[si];
        for (int k = 0; k < len - 1 && n < MAX_SEGMENTS; k++) {
            const double *p0 = sg->points + (start + k) * 3;
            const double *p1 = sg->points + (start + k + 1) * 3;
            seg6[n * 6 + 0] = p0[0]; seg6[n * 6 + 1] = p0[1]; seg6[n * 6 + 2] = p0[2];
            seg6[n * 6 + 3] = p1[0]; seg6[n * 6 + 4] = p1[1]; seg6[n * 6 + 5] = p1[2];
            meta3[n * 3 + 0] = si;
            meta3[n * 3 + 1] = pid[start + k];
            meta3[n * 3 + 2] = pid[start + k + 1];
            seg_loc2[n * 2 + 0] = si;
            seg_loc2[n * 2 + 1] = k;
            n++;
        }
    }
    return n;
}

/* Rotate all segments: out = seg @ rot^T. seg and out are [n][6]. */
static void rotate_segments(const double *seg, double *out, int n, const double *rot) {
    for (int i = 0; i < n; i++) {
        for (int pt = 0; pt < 2; pt++) {
            const double *p = seg + i * 6 + pt * 3;
            double *q = out + i * 6 + pt * 3;
            q[0] = p[0]*rot[0] + p[1]*rot[1] + p[2]*rot[2];
            q[1] = p[0]*rot[3] + p[1]*rot[4] + p[2]*rot[5];
            q[2] = p[0]*rot[6] + p[1]*rot[7] + p[2]*rot[8];
        }
    }
}

/* Build the diagram from crossings and resolve via Yamada skein.
 * Returns the Yamada polynomial. Sets *n_crossings_out. */
static LPoly build_and_resolve(const SpatialGraph *sg, const CrossingRecord *xings,
                                int n_xings, const double *rot_seg, const int *seg_loc,
                                int n_seg, int *n_crossings_out) {
    int nV = sg->n_vertices;
    *n_crossings_out = n_xings;

    /* Assign CCW ports for each crossing */
    int slot[MAX_CROSSINGS][4]; /* slot[k][port_label] = ccw_position */
    /* port_label: 0=over_out, 1=over_in, 2=under_out, 3=under_in */
    for (int k = 0; k < n_xings; k++) {
        int gi = xings[k].seg_i, gj = xings[k].seg_j;
        int over_first = xings[k].zi > xings[k].zj;
        const double *a = rot_seg + gi * 6;
        const double *b = rot_seg + gj * 6;
        double di[2] = {a[3]-a[0], a[4]-a[1]};
        double dj[2] = {b[3]-b[0], b[4]-b[1]};
        double *d_over = over_first ? di : dj;
        double *d_under = over_first ? dj : di;
        double dirs[4][2] = {
            { d_over[0],  d_over[1]},   /* over_out */
            {-d_over[0], -d_over[1]},   /* over_in */
            { d_under[0], d_under[1]},  /* under_out */
            {-d_under[0],-d_under[1]}   /* under_in */
        };
        double ang[4];
        for (int i = 0; i < 4; i++) ang[i] = atan2(dirs[i][1], dirs[i][0]);
        int order[4] = {0,1,2,3};
        /* Sort by angle */
        for (int i = 0; i < 3; i++)
            for (int j = i+1; j < 4; j++)
                if (ang[order[i]] > ang[order[j]]) {
                    int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                }
        /* Rotate so first is an over-strand port (label 0 or 1) */
        while (order[0] >= 2) {
            int tmp = order[0];
            order[0] = order[1]; order[1] = order[2];
            order[2] = order[3]; order[3] = tmp;
        }
        for (int pos = 0; pos < 4; pos++)
            slot[k][order[pos]] = pos;
    }

    /* Build chain of events per strand */
    typedef struct { int sort_key_seg; double sort_key_t; int xing_idx; int is_over; } StrandEvent;
    StrandEvent events[MAX_STRANDS][MAX_CROSSINGS];
    int n_events[MAX_STRANDS];
    memset(n_events, 0, sizeof(n_events));

    for (int k = 0; k < n_xings; k++) {
        int gi = xings[k].seg_i, gj = xings[k].seg_j;
        int si_strand = seg_loc[gi * 2], si_seg = seg_loc[gi * 2 + 1];
        int sj_strand = seg_loc[gj * 2], sj_seg = seg_loc[gj * 2 + 1];
        int over_first = xings[k].zi > xings[k].zj;

        int ni = n_events[si_strand]++;
        events[si_strand][ni].sort_key_seg = si_seg;
        events[si_strand][ni].sort_key_t = xings[k].ti;
        events[si_strand][ni].xing_idx = k;
        events[si_strand][ni].is_over = over_first ? 1 : 0;

        int nj = n_events[sj_strand]++;
        events[sj_strand][nj].sort_key_seg = sj_seg;
        events[sj_strand][nj].sort_key_t = xings[k].tj;
        events[sj_strand][nj].xing_idx = k;
        events[sj_strand][nj].is_over = over_first ? 0 : 1;
    }

    /* Sort events per strand by (segment, t) */
    for (int si = 0; si < sg->n_strands; si++) {
        int n = n_events[si];
        for (int i = 0; i < n-1; i++)
            for (int j = i+1; j < n; j++) {
                StrandEvent *a = &events[si][i], *b = &events[si][j];
                if (a->sort_key_seg > b->sort_key_seg ||
                    (a->sort_key_seg == b->sort_key_seg && a->sort_key_t > b->sort_key_t)) {
                    StrandEvent tmp = *a; *a = *b; *b = tmp;
                }
            }
    }

    /* Build diagram edges */
    DiagramEdge dedges[MAX_DIAGRAM_EDGES];
    int n_dedges = 0;
    for (int si = 0; si < sg->n_strands; si++) {
        /* Chain: V(start) -> X0 -> X1 -> ... -> V(end) */
        DiagramEndpoint prev;
        prev.is_vertex = 1; prev.id = sg->strand_endv[si * 2]; prev.port = 0;

        for (int ei = 0; ei < n_events[si] && n_dedges < MAX_DIAGRAM_EDGES; ei++) {
            StrandEvent *ev = &events[si][ei];
            int role = ev->is_over; /* 1=over, 0=under */
            /* Incoming port: over_in(1) or under_in(3) */
            int port_in = role ? slot[ev->xing_idx][1] : slot[ev->xing_idx][3];
            DiagramEndpoint cur;
            cur.is_vertex = 0; cur.id = ev->xing_idx; cur.port = port_in;
            dedges[n_dedges].a = prev;
            dedges[n_dedges].b = cur;
            n_dedges++;
            /* Outgoing: over_out(0) or under_out(2) */
            prev.is_vertex = 0; prev.id = ev->xing_idx;
            prev.port = role ? slot[ev->xing_idx][0] : slot[ev->xing_idx][2];
        }
        /* Final edge to end vertex */
        if (n_dedges < MAX_DIAGRAM_EDGES) {
            DiagramEndpoint end;
            end.is_vertex = 1; end.id = sg->strand_endv[si * 2 + 1]; end.port = 0;
            dedges[n_dedges].a = prev;
            dedges[n_dedges].b = end;
            n_dedges++;
        }
    }

    /* ---- Skein resolution: sum over 3^n_xings states ---- */
    long long n_states = 1;
    for (int i = 0; i < n_xings; i++) n_states *= 3;

    LPoly total = POLY_ZERO;
    for (long long state_id = 0; state_id < n_states; state_id++) {
        int st[MAX_CROSSINGS];
        long long sid = state_id;
        int exp = 0;
        for (int i = 0; i < n_xings; i++) {
            st[i] = (int)(sid % 3); sid /= 3;
        }
        for (int i = 0; i < n_xings; i++) {
            if (st[i] == 0) exp++;
            else if (st[i] == 1) exp--;
        }

        /* Build vertex mapping: xmap[k][port] -> vertex id */
        int xmap[MAX_CROSSINGS][4];
        int next_id = nV;
        for (int k = 0; k < n_xings; k++) {
            if (st[k] == 2) {
                /* 4-valent vertex */
                int v = next_id++;
                xmap[k][0] = xmap[k][1] = xmap[k][2] = xmap[k][3] = v;
            } else if (st[k] == 0) {
                /* A-smoothing: (0,3) and (1,2) */
                int v1 = next_id++, v2 = next_id++;
                xmap[k][0] = v1; xmap[k][3] = v1;
                xmap[k][1] = v2; xmap[k][2] = v2;
            } else {
                /* B-smoothing: (0,1) and (2,3) */
                int v1 = next_id++, v2 = next_id++;
                xmap[k][0] = v1; xmap[k][1] = v1;
                xmap[k][2] = v2; xmap[k][3] = v2;
            }
        }

        /* Build graph edges from diagram edges */
        Edge g_edges[MAX_DIAGRAM_EDGES];
        int ng = 0;
        for (int i = 0; i < n_dedges; i++) {
            int va, vb;
            if (dedges[i].a.is_vertex) va = dedges[i].a.id;
            else va = xmap[dedges[i].a.id][dedges[i].a.port];
            if (dedges[i].b.is_vertex) vb = dedges[i].b.id;
            else vb = xmap[dedges[i].b.id][dedges[i].b.port];
            g_edges[ng].u = va; g_edges[ng].v = vb; ng++;
        }

        LPoly val = graph_R(next_id, g_edges, ng);
        LPoly term = poly_scale_mono(&val, exp, 1);
        total = poly_add(&total, &term);
    }

    return total;
}

/* ========================================================================
 * 5. Public API: Yamada polynomial of a spatial graph
 * ======================================================================== */

/* Random rotation via QR decomposition (simplified: Householder). */
static void random_rotation(double *rot, unsigned long long *rng_state) {
    /* Simple xorshift64* RNG */
    double M[9];
    for (int i = 0; i < 9; i++) {
        *rng_state ^= *rng_state >> 12;
        *rng_state ^= *rng_state << 25;
        *rng_state ^= *rng_state >> 27;
        unsigned long long x = *rng_state * 0x2545F4914F6CDD1DULL;
        M[i] = ((double)(x >> 11)) / (double)(1ULL << 53) * 2.0 - 1.0;
    }
    /* Gram-Schmidt to get orthonormal basis */
    double *r0 = rot, *r1 = rot+3, *r2 = rot+6;
    double n;
    /* Row 0 */
    r0[0]=M[0]; r0[1]=M[1]; r0[2]=M[2];
    n = sqrt(r0[0]*r0[0]+r0[1]*r0[1]+r0[2]*r0[2]);
    if (n < 1e-10) { r0[0]=1; r0[1]=r0[2]=0; n=1; }
    r0[0]/=n; r0[1]/=n; r0[2]/=n;
    /* Row 1: orthogonalize to row 0 */
    r1[0]=M[3]; r1[1]=M[4]; r1[2]=M[5];
    double d = r1[0]*r0[0]+r1[1]*r0[1]+r1[2]*r0[2];
    r1[0]-=d*r0[0]; r1[1]-=d*r0[1]; r1[2]-=d*r0[2];
    n = sqrt(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2]);
    if (n < 1e-10) { r1[0]=0; r1[1]=1; r1[2]=0; n=1; }
    r1[0]/=n; r1[1]/=n; r1[2]/=n;
    /* Row 2: cross product */
    r2[0]=r0[1]*r1[2]-r0[2]*r1[1];
    r2[1]=r0[2]*r1[0]-r0[0]*r1[2];
    r2[2]=r0[0]*r1[1]-r0[1]*r1[0];
}

/* Compute Yamada polynomial of strands.
 * strands: array of 3D polyline points, concatenated.
 * strand_lens: number of points per strand.
 * n_strands: number of strands.
 * tol: endpoint coincidence tolerance.
 * tries: number of random projections to try.
 * seed: RNG seed.
 * result: output polynomial.
 * Returns number of crossings in best projection, or -1 on failure. */
int yamada_polynomial(const double *strands, const int *strand_lens, int n_strands,
                      double tol, int tries, unsigned long long seed,
                      LPoly *result, int *n_crossings_out) {
    /* Build spatial graph */
    SpatialGraph sg;
    sg.n_strands = n_strands;
    int total_pts = 0;
    for (int si = 0; si < n_strands; si++) {
        sg.strand_start[si] = total_pts;
        sg.strand_len[si] = strand_lens[si];
        for (int k = 0; k < strand_lens[si]; k++) {
            sg.points[(total_pts + k) * 3 + 0] = strands[(total_pts + k) * 3 + 0];
            sg.points[(total_pts + k) * 3 + 1] = strands[(total_pts + k) * 3 + 1];
            sg.points[(total_pts + k) * 3 + 2] = strands[(total_pts + k) * 3 + 2];
        }
        total_pts += strand_lens[si];
    }

    /* Identify graph vertices (coincident endpoints) */
    sg.n_vertices = 0;
    double verts[MAX_POINTS * 3];
    for (int si = 0; si < n_strands; si++) {
        for (int ep = 0; ep < 2; ep++) {
            int pidx = sg.strand_start[si] + (ep ? sg.strand_len[si] - 1 : 0);
            const double *p = sg.points + pidx * 3;
            int found = -1;
            for (int vi = 0; vi < sg.n_vertices; vi++) {
                double dx = p[0]-verts[vi*3], dy = p[1]-verts[vi*3+1], dz = p[2]-verts[vi*3+2];
                if (sqrt(dx*dx+dy*dy+dz*dz) < tol) { found = vi; break; }
            }
            if (found < 0) {
                found = sg.n_vertices;
                verts[found*3]=p[0]; verts[found*3+1]=p[1]; verts[found*3+2]=p[2];
                sg.n_vertices++;
            }
            sg.strand_endv[si * 2 + ep] = found;
        }
    }
    memcpy(sg.vertices, verts, (size_t)(sg.n_vertices * 3) * sizeof(double));

    /* Build segments */
    double seg6[MAX_SEGMENTS * 6];
    int meta3[MAX_SEGMENTS * 3], seg_loc2[MAX_SEGMENTS * 2];
    int n_seg = build_segments(&sg, tol, seg6, meta3, seg_loc2);

    /* Try projections */
    double rot_seg[MAX_SEGMENTS * 6];
    CrossingRecord xings[MAX_CROSSINGS];
    int best_nx = -1;
    CrossingRecord best_xings[MAX_CROSSINGS];
    double best_rot_seg[MAX_SEGMENTS * 6];
    unsigned long long rng = seed ? seed : 42;

    for (int trial = 0; trial < tries; trial++) {
        double rot[9];
        if (trial == 0) {
            /* Identity */
            memset(rot, 0, sizeof(rot));
            rot[0] = rot[4] = rot[8] = 1.0;
        } else {
            random_rotation(rot, &rng);
        }
        rotate_segments(seg6, rot_seg, n_seg, rot);

        /* Check graph vertices don't project onto crossings */
        int nx = find_crossings(rot_seg, meta3, n_seg, xings);
        if (nx < 0) continue;

        /* Check graph vertex proximity to crossings */
        int bad = 0;
        for (int vi = 0; vi < sg.n_vertices && !bad; vi++) {
            double *v = sg.vertices + vi * 3;
            double vr[2] = {
                v[0]*rot[0]+v[1]*rot[1]+v[2]*rot[2],
                v[0]*rot[3]+v[1]*rot[4]+v[2]*rot[5]
            };
            for (int k = 0; k < nx; k++) {
                double dx = vr[0]-xings[k].px, dy = vr[1]-xings[k].py;
                if (dx*dx+dy*dy < 1e-12) { bad = 1; break; }
            }
        }
        if (bad) continue;

        if (best_nx < 0 || nx < best_nx) {
            best_nx = nx;
            memcpy(best_xings, xings, (size_t)nx * sizeof(CrossingRecord));
            memcpy(best_rot_seg, rot_seg, (size_t)(n_seg * 6) * sizeof(double));
        }
        if (best_nx <= 2) break;
    }

    if (best_nx < 0) return -1;

    *n_crossings_out = best_nx;
    memo_clear();
    *result = build_and_resolve(&sg, best_xings, best_nx, best_rot_seg, seg_loc2, n_seg, n_crossings_out);
    return best_nx;
}

/* ========================================================================
 * 6. Ctypes entry points
 * ======================================================================== */

/* Compute Yamada polynomial for one set of strands.
 * Input: flat array of 3D points, strand lengths, parameters.
 * Output: polynomial coefficients written to out_coeff[POLY_SIZE],
 *         with out_coeff[i] = coefficient of A^(i-MAX_EXP).
 * Returns number of crossings, or -1 on failure. */
int compute_yamada(const double *strand_points, const int *strand_lens,
                   int n_strands, double tol, int tries, unsigned long long seed,
                   int *out_coeff, int *out_n_crossings) {
    LPoly poly;
    int nx;
    int ret = yamada_polynomial(strand_points, strand_lens, n_strands,
                                tol, tries, seed, &poly, &nx);
    if (ret < 0) {
        memset(out_coeff, 0, POLY_SIZE * sizeof(int));
        *out_n_crossings = 0;
        return -1;
    }
    LPoly norm = poly_normalize(&poly);
    memcpy(out_coeff, norm.coeff, POLY_SIZE * sizeof(int));
    *out_n_crossings = nx;
    return ret;
}

/* Get POLY_SIZE and OFFSET constants for the Python caller. */
int get_poly_size(void) { return POLY_SIZE; }
int get_offset(void) { return OFFSET; }

/* Clear the memoization cache. Call between cells. */
void clear_memo(void) { memo_clear(); }
