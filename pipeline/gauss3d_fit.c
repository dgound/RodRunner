/*
 * Fast 3D Gaussian PSF fitting using Levenberg-Marquardt.
 * Callable from Python via ctypes.
 *
 * Model: f(z,y,x) = A * exp(-((z-z0)^2/(2*sz^2) + (y-y0)^2/(2*sy^2) + (x-x0)^2/(2*sx^2))) + bg
 * Parameters: [A, z0, y0, x0, sz, sy, sx, bg]
 *
 * Return codes (from fit_gaussian_3d and propagated by batch_fit):
 *    0 = success: params + errors are valid.
 *    2 = params valid but error estimation failed (singular covariance).
 *        params_out[8..10] (z_err, y_err, x_err) are set to -1 as sentinels.
 *   -1 = fit failed: do not use params_out.
 *   -3 = STRICT CENTRE-OWNERSHIP failure (only reachable when strict_own != 0):
 *        no fit was obtained whose centre lies inside this cell's own mask.
 *        params_out must not be used. Distinct from -1 so callers can tell a
 *        numerically failed fit from an ownership rejection.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NPAR 8
#define MAX_ITER 100

/* Ascending-order comparator for qsort<double>. */
static int _cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Runtime OpenMP diagnostics — callable from Python via ctypes. Allow the
 * driver to report whether the loaded .so was compiled with OpenMP and the
 * thread count that will be used. No effect if OpenMP isn't available. */
/* Optional upper cap on sigma_z (axial PSF width), in voxels. <=0 disables
 * (default), preserving the original loose bound sigma_z in [0.2, nz]. When
 * set > 0, the axial width is clamped to [0.2, g_sigma_z_max] and the seed is
 * lowered to match. Set once from Python before batch_fit; read-only during
 * the OpenMP region, so no synchronization is needed. */
static double g_sigma_z_max = 0.0;

void set_sigma_z_max(double v) { g_sigma_z_max = v; }

int has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int openmp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static inline double model(double z, double y, double x, const double *p) {
    double dz = z - p[1], dy = y - p[2], dx = x - p[3];
    double sz2 = p[4]*p[4], sy2 = p[5]*p[5], sx2 = p[6]*p[6];
    return p[0] * exp(-0.5*(dz*dz/sz2 + dy*dy/sy2 + dx*dx/sx2)) + p[7];
}

/*
 * Fit one 3D Gaussian to a crop.
 *
 *   data: row-major [nz][ny][nx] float64
 *   mask: row-major [nz][ny][nx] uint8 (1=inside cell ROI, 0=outside)
 *   nz, ny, nx: dimensions
 *   params_out: [A, z0, y0, x0, sz, sy, sx, bg, z_err, y_err, x_err] (11 doubles)
 *
 * Masking semantics (IMPORTANT): all residuals, the Jacobian / JtJ, and the
 * final covariance are accumulated ONLY over masked voxels. Unmasked voxels
 * in the padded crop are excluded from the objective entirely — they're used
 * only as a background sample when there are enough of them.
 *
 * Background initialization:
 *   - If there are enough unmasked voxels in the crop (>= NPAR+1), use the
 *     median of unmasked voxel intensities. These voxels live in the padded
 *     region outside the cell and are the most defensible background sample.
 *   - Otherwise fall back to the 10th percentile of masked intensities.
 *
 * Fit objective is unweighted OLS (known simplification; does not weight by
 * photon shot noise).
 */
/* Nearest-voxel label under a candidate centre. Returns -1 when the centre is
 * outside the crop. Categorical: the label is looked up, never interpolated. */
static inline int center_label(const int *labels, int nz, int ny, int nx,
                               double cz_d, double cy_d, double cx_d) {
    int cz = (int)lround(cz_d);
    int cy = (int)lround(cy_d);
    int cx = (int)lround(cx_d);
    if (cz < 0 || cz >= nz || cy < 0 || cy >= ny || cx < 0 || cx >= nx)
        return -1;
    return labels[cz*ny*nx + cy*nx + cx];
}

/*
 * strict_own semantics (Requirement 3, constrained refit)
 * -------------------------------------------------------
 * strict_own == 0 reproduces the historical unconstrained fitter exactly: the
 * only centre restriction is the pre-existing neighbour guard (the centre may
 * not sit inside a DIFFERENT cell's label; background pad is allowed).
 *
 * strict_own != 0 additionally requires the centre to stay inside THIS cell's
 * own label at every accepted step:
 *   - the seed is the brightest owned voxel; if that voxel is not owned the fit
 *     is refused outright with -3 (no salvage, no search elsewhere);
 *   - a trial step whose nearest-voxel centre is not owned is REJECTED and the
 *     LM damping is increased, which shrinks the next proposed step. The
 *     coordinate is never clamped or projected onto the mask;
 *   - if no step is ever accepted, or the final centre is not owned, the fit
 *     returns -3 rather than a centre outside its support.
 */
int fit_gaussian_3d(const double *data, const unsigned char *mask,
                    const int *labels, int cell_id,
                    int nz, int ny, int nx, int strict_own, double *params_out) {
    int n = nz * ny * nx;
    int npix = 0;       /* # masked voxels (fit support) */
    int n_unmasked = 0; /* # unmasked voxels (background sample) */

    /* Strict mode seeds from the brightest OWNED voxel, which is not necessarily
     * the brightest masked voxel if a caller passes a mask broader than
     * labels==cell_id. Scanned here, BEFORE the generic "not enough support"
     * check, so a cell with no owned support at this timepoint (e.g. a z gate
     * that excluded every plane) is reported as an ownership refusal rather than
     * as a numerical fit failure. */
    double owned_peak = -1e30;
    int owned_z = -1, owned_y = -1, owned_x = -1;
    if (strict_own) {
        for (int iz = 0; iz < nz; iz++)
            for (int iy = 0; iy < ny; iy++)
                for (int ix = 0; ix < nx; ix++) {
                    int idx = iz*ny*nx + iy*nx + ix;
                    if (labels[idx] != cell_id) continue;
                    if (data[idx] > owned_peak) {
                        owned_peak = data[idx];
                        owned_z = iz; owned_y = iy; owned_x = ix;
                    }
                }
        if (owned_z < 0) return -3;
    }

    /* One pass: find peak location among masked pixels, collect masked and
     * unmasked intensity samples for the background estimator.
     */
    double peak_val = -1e30;
    int peak_z = 0, peak_y = 0, peak_x = 0;
    double *masked_vals   = (double *)malloc((size_t)n * sizeof(double));
    double *unmasked_vals = (double *)malloc((size_t)n * sizeof(double));
    if (!masked_vals || !unmasked_vals) {
        free(masked_vals); free(unmasked_vals);
        return -1;
    }

    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                int idx = iz*ny*nx + iy*nx + ix;
                double v = data[idx];
                if (mask[idx]) {
                    masked_vals[npix++] = v;
                    if (v > peak_val) {
                        peak_val = v;
                        peak_z = iz; peak_y = iy; peak_x = ix;
                    }
                } else {
                    unmasked_vals[n_unmasked++] = v;
                }
            }

    if (npix < NPAR + 1) {
        free(masked_vals); free(unmasked_vals);
        return -1;
    }

    double bg;
    if (n_unmasked >= NPAR + 1) {
        /* Preferred: median of unmasked voxels — this is literally the
         * background in the padded region outside the cell. */
        qsort(unmasked_vals, (size_t)n_unmasked, sizeof(double), _cmp_double);
        bg = unmasked_vals[n_unmasked / 2];
    } else {
        /* Fallback: not enough unmasked support in the crop. Use the 10th
         * percentile of masked voxels as a rough low-intensity estimator. */
        qsort(masked_vals, (size_t)npix, sizeof(double), _cmp_double);
        int p10_idx = (int)(0.10 * (npix - 1));
        bg = masked_vals[p10_idx];
    }
    free(masked_vals);
    free(unmasked_vals);

    if (strict_own) {
        peak_val = owned_peak;
        peak_z = owned_z; peak_y = owned_y; peak_x = owned_x;
    }

    double amp = peak_val - bg;
    if (amp <= 0) return -1;

    /* Initial params.
     *   sigma_z = 4.0 slices: matches an axial PSF of ~400 nm at a 100 nm
     *   z-step, which is the physically reasonable regime. The previous seed
     *   of 1.0 was 4x too small and made LM crawl out of a tight local basin.
     */
    double sz_seed = 4.0;
    if (g_sigma_z_max > 0 && g_sigma_z_max < sz_seed) sz_seed = g_sigma_z_max;
    double p[NPAR] = {amp, (double)peak_z, (double)peak_y, (double)peak_x, sz_seed, 1.5, 1.5, bg};

    /* Levenberg-Marquardt */
    double lambda = 0.001;
    double JtJ[NPAR*NPAR], Jtr[NPAR], J[NPAR];
    int n_accepted = 0;   /* strict mode refuses a fit that never took a step */

    for (int iter = 0; iter < MAX_ITER; iter++) {
        memset(JtJ, 0, sizeof(JtJ));
        memset(Jtr, 0, sizeof(Jtr));
        double chi2 = 0;

        /* Objective / Jacobian accumulated ONLY over masked voxels. */
        for (int iz = 0; iz < nz; iz++)
            for (int iy = 0; iy < ny; iy++)
                for (int ix = 0; ix < nx; ix++) {
                    int idx = iz*ny*nx + iy*nx + ix;
                    if (!mask[idx]) continue;
                    double d = data[idx];
                    double z = (double)iz, y = (double)iy, x = (double)ix;

                    double dz = z - p[1], dy = y - p[2], dx = x - p[3];
                    double sz2 = p[4]*p[4], sy2 = p[5]*p[5], sx2 = p[6]*p[6];
                    double exponent = -0.5*(dz*dz/sz2 + dy*dy/sy2 + dx*dx/sx2);
                    double e = exp(exponent);
                    double f = p[0] * e + p[7];
                    double r = d - f;
                    chi2 += r * r;

                    /* Jacobian */
                    J[0] = e;                              /* dF/dA */
                    J[1] = p[0] * e * dz / sz2;           /* dF/dz0 */
                    J[2] = p[0] * e * dy / sy2;           /* dF/dy0 */
                    J[3] = p[0] * e * dx / sx2;           /* dF/dx0 */
                    J[4] = p[0] * e * dz*dz / (p[4]*sz2); /* dF/dsz */
                    J[5] = p[0] * e * dy*dy / (p[5]*sy2); /* dF/dsy */
                    J[6] = p[0] * e * dx*dx / (p[6]*sx2); /* dF/dsx */
                    J[7] = 1.0;                            /* dF/dbg */

                    for (int a = 0; a < NPAR; a++) {
                        Jtr[a] += J[a] * r;
                        for (int b = 0; b < NPAR; b++)
                            JtJ[a*NPAR+b] += J[a] * J[b];
                    }
                }

        /* Damped normal equations: (JtJ + lambda*diag(JtJ)) * dp = Jtr */
        double A[NPAR*NPAR], b[NPAR];
        memcpy(A, JtJ, sizeof(A));
        memcpy(b, Jtr, sizeof(b));
        for (int i = 0; i < NPAR; i++)
            A[i*NPAR+i] *= (1.0 + lambda);

        /* Solve via Gaussian elimination (small 8x8 system) */
        double aug[NPAR][NPAR+1];
        for (int i = 0; i < NPAR; i++) {
            for (int j = 0; j < NPAR; j++)
                aug[i][j] = A[i*NPAR+j];
            aug[i][NPAR] = b[i];
        }
        for (int col = 0; col < NPAR; col++) {
            /* Partial pivoting */
            int maxrow = col;
            for (int row = col+1; row < NPAR; row++)
                if (fabs(aug[row][col]) > fabs(aug[maxrow][col]))
                    maxrow = row;
            if (maxrow != col) {
                double tmp[NPAR+1];
                memcpy(tmp, aug[col], sizeof(tmp));
                memcpy(aug[col], aug[maxrow], sizeof(tmp));
                memcpy(aug[maxrow], tmp, sizeof(tmp));
            }
            if (fabs(aug[col][col]) < 1e-15) return -1;
            for (int row = col+1; row < NPAR; row++) {
                double f = aug[row][col] / aug[col][col];
                for (int j = col; j <= NPAR; j++)
                    aug[row][j] -= f * aug[col][j];
            }
        }
        double dp[NPAR];
        for (int i = NPAR-1; i >= 0; i--) {
            dp[i] = aug[i][NPAR];
            for (int j = i+1; j < NPAR; j++)
                dp[i] -= aug[i][j] * dp[j];
            dp[i] /= aug[i][i];
        }

        /* Trial update */
        double p_new[NPAR];
        for (int i = 0; i < NPAR; i++)
            p_new[i] = p[i] + dp[i];

        /* Amplitude: if the step drives A non-positive, reject the step
         * entirely and increase lambda (same as a chi2-increasing step).
         * Clamping A to zero would zero all position/width Jacobian entries,
         * making JtJ rank-2 and triggering the singularity guard. */
        if (p_new[0] <= 0) {
            lambda *= 10.0;
            if (lambda > 1e10) break;
            continue;
        }
        /* Center must stay inside the crop */
        if (p_new[1] < 0) p_new[1] = 0;
        if (p_new[1] > nz - 1) p_new[1] = nz - 1;
        if (p_new[2] < 0) p_new[2] = 0;
        if (p_new[2] > ny - 1) p_new[2] = ny - 1;
        if (p_new[3] < 0) p_new[3] = 0;
        if (p_new[3] > nx - 1) p_new[3] = nx - 1;
        /* Sigma bounds */
        if (p_new[4] < 0.2) p_new[4] = 0.2;
        double sz_hi = (g_sigma_z_max > 0) ? g_sigma_z_max : (double)nz;
        if (sz_hi < 0.2) sz_hi = 0.2;   /* keep the cap >= the 0.2 floor for a degenerate sigma_z_max */
        if (p_new[4] > sz_hi) p_new[4] = sz_hi;
        if (p_new[5] < 0.3) p_new[5] = 0.3;
        if (p_new[5] > ny / 2.0) p_new[5] = ny / 2.0;
        if (p_new[6] < 0.3) p_new[6] = 0.3;
        if (p_new[6] > nx / 2.0) p_new[6] = nx / 2.0;
        if (p_new[7] < 0) p_new[7] = 0;

        /* Targeted guard: forbid the centre from sitting inside a DIFFERENT
         * cell's mask (a neighbour-grab). Background pad (label 0) and the own
         * cell are allowed; this leaves legitimate peripheral loci untouched. */
        int center_in_other = 0;
        int center_not_owned = 0;
        {
            int lv = center_label(labels, nz, ny, nx,
                                  p_new[1], p_new[2], p_new[3]);
            if (lv > 0 && lv != cell_id) center_in_other = 1;
            /* Strict mode: background (0) and off-crop (-1) are rejected too. */
            if (lv != cell_id) center_not_owned = 1;
        }
        int center_rejected = center_in_other || (strict_own && center_not_owned);

        /* Trial chi2: masked voxels only, matching the main LM objective. */
        double chi2_new = 0;
        for (int iz = 0; iz < nz; iz++)
            for (int iy = 0; iy < ny; iy++)
                for (int ix = 0; ix < nx; ix++) {
                    int idx = iz*ny*nx + iy*nx + ix;
                    if (!mask[idx]) continue;
                    double f = model((double)iz, (double)iy, (double)ix, p_new);
                    double r = data[idx] - f;
                    chi2_new += r * r;
                }

        if (chi2_new < chi2 && !center_rejected) {
            memcpy(p, p_new, sizeof(p));
            n_accepted++;
            lambda *= 0.1;
            if (fabs(chi2 - chi2_new) / (chi2 + 1e-30) < 1e-6)
                break;
        } else {
            /* Increasing lambda shrinks the next proposed step: an out-of-support
             * proposal makes the optimizer try a smaller move, it does not move
             * the coordinate onto the mask. */
            lambda *= 10.0;
            if (lambda > 1e10) break;
        }
    }

    /* Strict mode: refuse rather than return a centre outside its own support,
     * and refuse a "fit" that is really just the untouched integer seed. */
    if (strict_own) {
        if (n_accepted == 0) return -3;
        if (center_label(labels, nz, ny, nx, p[1], p[2], p[3]) != cell_id)
            return -3;
    }

    /* Compute parameter errors from inverse of JtJ (recompute at final params).
     * Masked voxels only, matching the main LM objective. */
    memset(JtJ, 0, sizeof(JtJ));
    double chi2_final = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                int idx = iz*ny*nx + iy*nx + ix;
                if (!mask[idx]) continue;
                double z = (double)iz, y = (double)iy, x = (double)ix;
                double dz = z - p[1], dy = y - p[2], dx = x - p[3];
                double sz2 = p[4]*p[4], sy2 = p[5]*p[5], sx2 = p[6]*p[6];
                double e = exp(-0.5*(dz*dz/sz2 + dy*dy/sy2 + dx*dx/sx2));
                double f = p[0]*e + p[7];
                double r = data[idx] - f;
                chi2_final += r*r;

                J[0] = e;
                J[1] = p[0]*e*dz/sz2;
                J[2] = p[0]*e*dy/sy2;
                J[3] = p[0]*e*dx/sx2;
                J[4] = p[0]*e*dz*dz/(p[4]*sz2);
                J[5] = p[0]*e*dy*dy/(p[5]*sy2);
                J[6] = p[0]*e*dx*dx/(p[6]*sx2);
                J[7] = 1.0;

                for (int a = 0; a < NPAR; a++)
                    for (int b = 0; b < NPAR; b++)
                        JtJ[a*NPAR+b] += J[a]*J[b];
            }

    /* Invert JtJ for covariance (8x8 — use augmented matrix). If the
     * Gaussian-Jordan step encounters a near-zero pivot, mark errors as the
     * -1 sentinel and set ret_code = 2 (params still valid, errors are not).
     */
    int ret_code = 0;
    double inv[NPAR][2*NPAR];
    for (int i = 0; i < NPAR; i++) {
        for (int j = 0; j < NPAR; j++) {
            inv[i][j] = JtJ[i*NPAR+j];
            inv[i][NPAR+j] = (i==j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < NPAR && ret_code == 0; col++) {
        int maxrow = col;
        for (int row = col+1; row < NPAR; row++)
            if (fabs(inv[row][col]) > fabs(inv[maxrow][col]))
                maxrow = row;
        if (maxrow != col) {
            double tmp[2*NPAR];
            memcpy(tmp, inv[col], sizeof(tmp));
            memcpy(inv[col], inv[maxrow], sizeof(tmp));
            memcpy(inv[maxrow], tmp, sizeof(tmp));
        }
        double piv = inv[col][col];
        if (fabs(piv) < 1e-30) {
            params_out[8] = params_out[9] = params_out[10] = -1;
            ret_code = 2;
            break;
        }
        for (int j = 0; j < 2*NPAR; j++)
            inv[col][j] /= piv;
        for (int row = 0; row < NPAR; row++) {
            if (row == col) continue;
            double f = inv[row][col];
            for (int j = 0; j < 2*NPAR; j++)
                inv[row][j] -= f * inv[col][j];
        }
    }

    if (ret_code == 0) {
        /* Degrees of freedom = # fitted voxels (masked) minus parameter count.
         * If support is too small, mark errors as the -1 sentinel. */
        int dof = npix - NPAR;
        if (dof <= 0) {
            params_out[8] = params_out[9] = params_out[10] = -1;
            ret_code = 2;
        } else {
            double s2 = chi2_final / (double)dof;
            if (s2 < 0) s2 = 0;
            params_out[8]  = sqrt(fabs(inv[1][NPAR+1] * s2)); /* z_err */
            params_out[9]  = sqrt(fabs(inv[2][NPAR+2] * s2)); /* y_err */
            params_out[10] = sqrt(fabs(inv[3][NPAR+3] * s2)); /* x_err */
        }
    }

    for (int i = 0; i < NPAR; i++)
        params_out[i] = p[i];

    return ret_code;
}

/*
 * Batch fit: fit all cells for one image.
 *
 * data:        full image [nz][ny][nx] float64
 * mask_labels: full mask  [nz][ny][nx] int32
 * bboxes:      [n_cells][6] int32 (z_min, z_max, y_min, y_max, x_min, x_max)
 * cell_ids:    [n_cells] int32, the mask label for each entry, or NULL for the
 *              historical implicit mapping cell_id = index + 1 (labels 1..n
 *              contiguous). An explicit list lets a caller re-fit an arbitrary
 *              SUBSET of cells without rebuilding a dense n_cells array.
 * strict_own:  0 = historical unconstrained fit (see fit_gaussian_3d).
 *              non-zero = constrained: the centre must remain inside its own
 *              label; otherwise the cell returns -3.
 * results:     [n_cells][11] float64 output
 * success:     [n_cells] int32 output. Values:
 *                  0 = success with valid params + errors
 *                  2 = params valid, errors not estimable (singular covariance)
 *                 -1 = fit failed; do not use params
 *                 -3 = strict centre-ownership failure (strict_own only)
 *
 * Threading: when compiled with OpenMP, the outer loop over cells is executed
 * in parallel. Each thread holds its own crop/cmask scratch buffers sized to
 * the largest bbox. All writes to results[c*11+...] and success[c] land at
 * disjoint addresses indexed by c, so no synchronization is needed.
 * fit_gaussian_3d itself uses only stack locals and per-call mallocs, so it
 * is thread-safe when called independently from multiple threads.
 */
void batch_fit_ex(const double *data, const int *mask_labels,
                  int nz, int ny, int nx,
                  const int *bboxes, int n_cells,
                  const int *cell_ids, int strict_own,
                  double *results, int *success) {

    /* nz is part of the documented ABI (callers pass the full volume shape) but
     * crop indexing needs only ny/nx. Marked used so -Wextra builds stay clean. */
    (void)nz;

    /* Find the largest crop needed once so each thread sizes its scratch
     * buffers accordingly. Serial pass — cheap compared to the fit itself. */
    int max_cn = 0;
    for (int c = 0; c < n_cells; c++) {
        int bz0 = bboxes[c*6+0], bz1 = bboxes[c*6+1];
        int by0 = bboxes[c*6+2], by1 = bboxes[c*6+3];
        int bx0 = bboxes[c*6+4], bx1 = bboxes[c*6+5];
        int cnz = bz1 - bz0 + 1;
        int cny = by1 - by0 + 1;
        int cnx = bx1 - bx0 + 1;
        int cn = cnz * cny * cnx;
        if (cn > max_cn) max_cn = cn;
    }
    if (max_cn <= 0) return;

    #pragma omp parallel
    {
        /* Thread-local scratch buffers: one allocation per thread, reused
         * across every iteration assigned to that thread. */
        double *crop         = (double *)malloc((size_t)max_cn * sizeof(double));
        unsigned char *cmask = (unsigned char *)malloc((size_t)max_cn);
        int *clabel          = (int *)malloc((size_t)max_cn * sizeof(int));
        int alloc_ok = (crop != NULL && cmask != NULL && clabel != NULL);

        /* schedule(dynamic, 16): cells have different bbox sizes → different
         * fit cost, so dynamic dispatch keeps threads from finishing early
         * while others still have a large cell to fit. Chunk of 16 avoids
         * per-iteration scheduling overhead. */
        #pragma omp for schedule(dynamic, 16)
        for (int c = 0; c < n_cells; c++) {
            if (!alloc_ok) {
                success[c] = -1;
                continue;
            }

            int cell_id = cell_ids ? cell_ids[c] : c + 1;
            int bz0 = bboxes[c*6+0], bz1 = bboxes[c*6+1];
            int by0 = bboxes[c*6+2], by1 = bboxes[c*6+3];
            int bx0 = bboxes[c*6+4], bx1 = bboxes[c*6+5];

            int cnz = bz1 - bz0 + 1;
            int cny = by1 - by0 + 1;
            int cnx = bx1 - bx0 + 1;

            for (int iz = 0; iz < cnz; iz++)
                for (int iy = 0; iy < cny; iy++)
                    for (int ix = 0; ix < cnx; ix++) {
                        int gi = (bz0+iz)*ny*nx + (by0+iy)*nx + (bx0+ix);
                        int li = iz*cny*cnx + iy*cnx + ix;
                        crop[li] = data[gi];
                        cmask[li] = (mask_labels[gi] == cell_id) ? 1 : 0;
                        clabel[li] = mask_labels[gi];
                    }

            int ret = fit_gaussian_3d(crop, cmask, clabel, cell_id,
                                      cnz, cny, cnx, strict_own, &results[c*11]);
            success[c] = ret;

            /* Convert local coords to global for success and "errors-only" failure. */
            if (ret == 0 || ret == 2) {
                results[c*11+1] += bz0; /* z0 */
                results[c*11+2] += by0; /* y0 */
                results[c*11+3] += bx0; /* x0 */
            }
        }

        free(crop);
        free(cmask);
        free(clabel);
    }
}

/* Historical entry point: dense labels 1..n_cells, unconstrained centre.
 * Kept byte-compatible so existing ctypes signatures need no change. */
void batch_fit(const double *data, const int *mask_labels,
               int nz, int ny, int nx,
               const int *bboxes, int n_cells,
               double *results, int *success) {
    batch_fit_ex(data, mask_labels, nz, ny, nx, bboxes, n_cells,
                 NULL, 0, results, success);
}
