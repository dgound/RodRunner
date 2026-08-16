/*
 * Replication-QC heavy inner-loop kernels, moved from Python/NumPy/SciPy to C.
 * Called from replication_qc.py via ctypes.
 *
 * The batch path performs a common-support model comparison:
 *   fit_1spot()         - refit M1 using the saved localization as its seed
 *   stage_a_and_sse1()  - residual diagnostics from the refitted M1
 *   fit_2spot()         - Levenberg-Marquardt fit of the constrained 2-spot
 *                         model (shared sigma_{z,y,x} and background, separate
 *                         amplitudes + centers), bounds enforced by projection.
 *
 * Not in C: file/path orchestration, CSV I/O, argparse, QC plotting.
 *
 * Parameter layout for the 2-spot model (matches replication_qc.py):
 *   p[ 0] = A1       p[ 1] = A2
 *   p[ 2] = cz1      p[ 3] = cy1      p[ 4] = cx1
 *   p[ 5] = cz2      p[ 6] = cy2      p[ 7] = cx2
 *   p[ 8] = sigma_z  p[ 9] = sigma_y  p[10] = sigma_x
 *   p[11] = bg
 *
 * Return codes from fit_2spot:
 *    0 = converged,            params + sse valid
 *    1 = not converged (hit max_iter or lambda overflow),
 *                              params + sse still returned (last good state)
 *    2 = non-finite params (invalid)
 *   -1 = invalid input (n_vox too small)
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NPAR 12
#define NPAR1 8
#define MAX_ITER_DEFAULT 200

/* Runtime OpenMP diagnostics — same pattern as gauss3d_fit.c so the Python
 * driver can report whether the loaded .so is parallel. */
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

/* Forward decls so the batch entry can call the existing public kernels. */
void stage_a_and_sse1(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    double amp1, double cz1, double cy1, double cx1,
    double sz1, double sy1, double sx1, double bg1,
    double *out_resid_score,
    double *out_peak_rel_amp,
    double *out_peak_offset_px,
    double *out_peak_z,
    double *out_peak_y,
    double *out_peak_x,
    double *out_sse1);

int fit_2spot(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    const double *p0, const double *lo, const double *hi,
    int max_iter,
    double *out_params, double *out_sse,
    int *out_pinned, int *out_nfev);


static inline double sqr(double x) { return x * x; }

static int center_supported(double cz, double cy, double cx,
                            const double *Z, const double *Y, const double *X,
                            int n_vox)
{
    int iz = (int)lround(cz), iy = (int)lround(cy), ix = (int)lround(cx);
    for (int i = 0; i < n_vox; i++) {
        if ((int)Z[i] == iz && (int)Y[i] == iy && (int)X[i] == ix) return 1;
    }
    return 0;
}

static inline double gauss3d(double amp,
                             double cz, double cy, double cx,
                             double sz, double sy, double sx,
                             double z,  double y,  double x)
{
    double dz = (z - cz) / sz;
    double dy = (y - cy) / sy;
    double dx = (x - cx) / sx;
    return amp * exp(-0.5 * (dz*dz + dy*dy + dx*dx));
}


/*
 * Single pass over the masked crop:
 *   - evaluate 1-spot model + background at every voxel
 *   - accumulate sum(model), sum(positive residual), 1-spot SSE
 *   - track max positive residual and its (z, y, x) location
 *
 * Python call sites get exactly the numbers they used to compute via NumPy,
 * without materializing the model and residual arrays.
 */
void stage_a_and_sse1(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    double amp1, double cz1, double cy1, double cx1,
    double sz1, double sy1, double sx1, double bg1,
    double *out_resid_score,
    double *out_peak_rel_amp,
    double *out_peak_offset_px,
    double *out_peak_z,
    double *out_peak_y,
    double *out_peak_x,
    double *out_sse1)
{
    double sum_pos   = 0.0;
    double sum_model = 0.0;
    double sse       = 0.0;
    double peak_val  = 0.0;
    int    peak_idx  = -1;

    for (int i = 0; i < n_vox; i++) {
        double g = gauss3d(amp1, cz1, cy1, cx1, sz1, sy1, sx1,
                           Z[i], Y[i], X[i]);
        double f = g + bg1;
        double r = obs[i] - f;
        sse       += r * r;
        sum_model += f;
        if (r > 0.0) {
            sum_pos += r;
            if (r > peak_val) { peak_val = r; peak_idx = i; }
        }
    }

    *out_sse1        = sse;
    *out_resid_score = sum_pos / (sum_model + 1e-9);

    if (peak_idx < 0) {
        *out_peak_rel_amp    = 0.0;
        *out_peak_offset_px  = INFINITY;
        *out_peak_z          = cz1;
        *out_peak_y          = cy1;
        *out_peak_x          = cx1;
    } else {
        double pz = Z[peak_idx], py = Y[peak_idx], px = X[peak_idx];
        *out_peak_rel_amp    = peak_val / (fabs(amp1) + 1e-9);
        *out_peak_z          = pz;
        *out_peak_y          = py;
        *out_peak_x          = px;
        *out_peak_offset_px  = sqrt(sqr(pz - cz1) + sqr(py - cy1) + sqr(px - cx1));
    }
}


/*
 * Compute chi2 (= SSE) for a given parameter vector. Used to evaluate the
 * trial step in the LM loop.
 */
static double chi2_2spot(const double *p,
                         const double *obs,
                         const double *Z, const double *Y, const double *X,
                         int n_vox)
{
    double A1 = p[0], A2 = p[1];
    double cz1 = p[2], cy1 = p[3], cx1 = p[4];
    double cz2 = p[5], cy2 = p[6], cx2 = p[7];
    double sz  = p[8], sy  = p[9], sx  = p[10], bg = p[11];
    double sz2 = sz*sz, sy2 = sy*sy, sx2 = sx*sx;

    if (!center_supported(cz1, cy1, cx1, Z, Y, X, n_vox)
        || !center_supported(cz2, cy2, cx2, Z, Y, X, n_vox)) return INFINITY;

    double chi2 = 0.0;
    for (int i = 0; i < n_vox; i++) {
        double z = Z[i], y = Y[i], x = X[i];
        double dz1 = z - cz1, dy1 = y - cy1, dx1 = x - cx1;
        double dz2 = z - cz2, dy2 = y - cy2, dx2 = x - cx2;
        double e1 = exp(-0.5 * (dz1*dz1/sz2 + dy1*dy1/sy2 + dx1*dx1/sx2));
        double e2 = exp(-0.5 * (dz2*dz2/sz2 + dy2*dy2/sy2 + dx2*dx2/sx2));
        double f  = A1 * e1 + A2 * e2 + bg;
        double r  = obs[i] - f;
        chi2 += r * r;
    }
    return chi2;
}


/*
 * Enforce box bounds on p (projection): p[i] = clamp(p[i], lo[i], hi[i])
 * when the bounds are finite. Infinite bounds skipped.
 */
static void clamp_bounds(double *p, const double *lo, const double *hi)
{
    for (int i = 0; i < NPAR; i++) {
        if (isfinite(lo[i]) && p[i] < lo[i]) p[i] = lo[i];
        if (isfinite(hi[i]) && p[i] > hi[i]) p[i] = hi[i];
    }
}

static int solve_linear(const double *a, const double *b, int n, double *x)
{
    double aug[NPAR][NPAR + 1];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = a[i * n + j];
        aug[i][n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int maxrow = col;
        for (int row = col + 1; row < n; row++)
            if (fabs(aug[row][col]) > fabs(aug[maxrow][col])) maxrow = row;
        if (maxrow != col) {
            for (int j = col; j <= n; j++) {
                double tmp = aug[col][j];
                aug[col][j] = aug[maxrow][j];
                aug[maxrow][j] = tmp;
            }
        }
        if (fabs(aug[col][col]) < 1e-15) return 0;
        for (int row = col + 1; row < n; row++) {
            double fac = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++) aug[row][j] -= fac * aug[col][j];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
        x[i] /= aug[i][i];
    }
    return 1;
}

static int invert_matrix(const double *a, int n, double *inv)
{
    double aug[NPAR][2 * NPAR];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = a[i * n + j];
            aug[i][n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < n; col++) {
        int maxrow = col;
        for (int row = col + 1; row < n; row++)
            if (fabs(aug[row][col]) > fabs(aug[maxrow][col])) maxrow = row;
        if (maxrow != col) {
            for (int j = 0; j < 2 * n; j++) {
                double tmp = aug[col][j];
                aug[col][j] = aug[maxrow][j];
                aug[maxrow][j] = tmp;
            }
        }
        double piv = aug[col][col];
        if (fabs(piv) < 1e-15) return 0;
        for (int j = 0; j < 2 * n; j++) aug[col][j] /= piv;
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double fac = aug[row][col];
            for (int j = 0; j < 2 * n; j++) aug[row][j] -= fac * aug[col][j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) inv[i * n + j] = aug[i][n + j];
    return 1;
}

static double chi2_1spot(const double *p,
                         const double *obs,
                         const double *Z, const double *Y, const double *X,
                         int n_vox)
{
    if (!center_supported(p[1], p[2], p[3], Z, Y, X, n_vox)) return INFINITY;
    double chi2 = 0.0;
    for (int i = 0; i < n_vox; i++) {
        double f = gauss3d(p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                           Z[i], Y[i], X[i]) + p[7];
        double r = obs[i] - f;
        chi2 += r * r;
    }
    return chi2;
}

int fit_1spot(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    const double *p0, const double *lo, const double *hi,
    int max_iter,
    double *out_params, double *out_sse, int *out_nfev)
{
    if (n_vox < NPAR1 + 1) return -1;
    if (max_iter <= 0) max_iter = MAX_ITER_DEFAULT;
    double p[NPAR1];
    memcpy(p, p0, sizeof(p));
    for (int i = 0; i < NPAR1; i++) {
        if (isfinite(lo[i]) && p[i] < lo[i]) p[i] = lo[i];
        if (isfinite(hi[i]) && p[i] > hi[i]) p[i] = hi[i];
    }
    double chi2 = chi2_1spot(p, obs, Z, Y, X, n_vox);
    if (!isfinite(chi2)) return -1;
    int nfev = 1, converged = 0;
    double lambda = 1e-3;
    for (int iter = 0; iter < max_iter; iter++) {
        double jtj[NPAR1 * NPAR1] = {0};
        double jtr[NPAR1] = {0};
        for (int i = 0; i < n_vox; i++) {
            double dz = Z[i] - p[1], dy = Y[i] - p[2], dx = X[i] - p[3];
            double sz2 = p[4]*p[4], sy2 = p[5]*p[5], sx2 = p[6]*p[6];
            double e = exp(-0.5 * (dz*dz/sz2 + dy*dy/sy2 + dx*dx/sx2));
            double g = p[0] * e;
            double r = obs[i] - (g + p[7]);
            double J[NPAR1] = {
                e, g*dz/sz2, g*dy/sy2, g*dx/sx2,
                g*dz*dz/(p[4]*sz2), g*dy*dy/(p[5]*sy2),
                g*dx*dx/(p[6]*sx2), 1.0,
            };
            for (int a = 0; a < NPAR1; a++) {
                jtr[a] += J[a] * r;
                for (int b = 0; b < NPAR1; b++) jtj[a*NPAR1+b] += J[a]*J[b];
            }
        }
        double grad_max = 0.0;
        for (int i = 0; i < NPAR1; i++) grad_max = fmax(grad_max, fabs(jtr[i]));
        if (grad_max <= 1e-8 * (1.0 + sqrt(fmax(chi2, 0.0)))) {
            converged = 1;
            break;
        }
        double a[NPAR1 * NPAR1];
        memcpy(a, jtj, sizeof(a));
        for (int i = 0; i < NPAR1; i++) a[i*NPAR1+i] *= (1.0 + lambda);
        double dp[NPAR1];
        if (!solve_linear(a, jtr, NPAR1, dp)) {
            lambda *= 10.0;
            if (lambda > 1e10) break;
            continue;
        }
        double p_new[NPAR1], step_max = 0.0;
        for (int i = 0; i < NPAR1; i++) {
            p_new[i] = p[i] + dp[i];
            if (isfinite(lo[i]) && p_new[i] < lo[i]) p_new[i] = lo[i];
            if (isfinite(hi[i]) && p_new[i] > hi[i]) p_new[i] = hi[i];
            step_max = fmax(step_max, fabs(p_new[i] - p[i])
                            / (1.0 + fabs(p[i])));
        }
        double chi2_new = chi2_1spot(p_new, obs, Z, Y, X, n_vox);
        nfev++;
        double tol = 1e-12 * (1.0 + fabs(chi2));
        if (isfinite(chi2_new) && chi2_new < chi2) {
            double rel = (chi2 - chi2_new) / (fabs(chi2) + 1e-30);
            memcpy(p, p_new, sizeof(p));
            chi2 = chi2_new;
            lambda *= 0.1;
            if (rel < 1e-6 || step_max < 1e-7) { converged = 1; break; }
        } else if (isfinite(chi2_new) && fabs(chi2_new - chi2) <= tol
                   && step_max < 1e-7) {
            converged = 1;
            break;
        } else {
            lambda *= 10.0;
            if (lambda > 1e10) break;
        }
    }
    for (int i = 0; i < NPAR1; i++) if (!isfinite(p[i])) return 2;
    memcpy(out_params, p, sizeof(p));
    *out_sse = chi2;
    *out_nfev = nfev;
    return converged ? 0 : 1;
}

static int fit2_uncertainty(
    const double *Z, const double *Y, const double *X,
    int n_vox, const double *p, double sse,
    double z_step_nm, double xy_pixel_nm,
    double *out_sigma_z_se, double *out_sep_se, double *out_sep_xy_se)
{
    if (n_vox <= NPAR) return 0;
    double jtj[NPAR * NPAR] = {0};
    double A1 = p[0], A2 = p[1];
    double sz = p[8], sy = p[9], sx = p[10];
    double sz2 = sz*sz, sy2 = sy*sy, sx2 = sx*sx;
    double sz3 = sz*sz2, sy3 = sy*sy2, sx3 = sx*sx2;
    for (int i = 0; i < n_vox; i++) {
        double dz1 = Z[i] - p[2], dy1 = Y[i] - p[3], dx1 = X[i] - p[4];
        double dz2 = Z[i] - p[5], dy2 = Y[i] - p[6], dx2 = X[i] - p[7];
        double e1 = exp(-0.5*(dz1*dz1/sz2 + dy1*dy1/sy2 + dx1*dx1/sx2));
        double e2 = exp(-0.5*(dz2*dz2/sz2 + dy2*dy2/sy2 + dx2*dx2/sx2));
        double g1 = A1*e1, g2 = A2*e2;
        double J[NPAR] = {
            e1, e2,
            g1*dz1/sz2, g1*dy1/sy2, g1*dx1/sx2,
            g2*dz2/sz2, g2*dy2/sy2, g2*dx2/sx2,
            (g1*dz1*dz1 + g2*dz2*dz2)/sz3,
            (g1*dy1*dy1 + g2*dy2*dy2)/sy3,
            (g1*dx1*dx1 + g2*dx2*dx2)/sx3,
            1.0,
        };
        for (int a = 0; a < NPAR; a++)
            for (int b = 0; b < NPAR; b++) jtj[a*NPAR+b] += J[a]*J[b];
    }
    double inv[NPAR * NPAR];
    if (!invert_matrix(jtj, NPAR, inv)) return 0;
    double s2 = sse / (double)(n_vox - NPAR);
    if (!(s2 >= 0) || !isfinite(s2)) return 0;
    for (int i = 0; i < NPAR*NPAR; i++) inv[i] *= s2;
    if (!(inv[8*NPAR+8] >= 0) || !isfinite(inv[8*NPAR+8])) return 0;
    *out_sigma_z_se = sqrt(inv[8*NPAR+8]);

    double dz = (p[2] - p[5]) * z_step_nm;
    double dy = (p[3] - p[6]) * xy_pixel_nm;
    double dx = (p[4] - p[7]) * xy_pixel_nm;
    double sep = sqrt(dz*dz + dy*dy + dx*dx);
    double sep_xy = sqrt(dy*dy + dx*dx);
    double g3[6] = {0}, gxy[6] = {0};
    if (sep > 0) {
        g3[0] =  dz*z_step_nm/sep; g3[3] = -g3[0];
        g3[1] =  dy*xy_pixel_nm/sep; g3[4] = -g3[1];
        g3[2] =  dx*xy_pixel_nm/sep; g3[5] = -g3[2];
    }
    if (sep_xy > 0) {
        gxy[1] =  dy*xy_pixel_nm/sep_xy; gxy[4] = -gxy[1];
        gxy[2] =  dx*xy_pixel_nm/sep_xy; gxy[5] = -gxy[2];
    }
    int idx[6] = {2,3,4,5,6,7};
    double var3 = 0.0, varxy = 0.0;
    for (int a = 0; a < 6; a++) {
        for (int b = 0; b < 6; b++) {
            double cv = inv[idx[a]*NPAR + idx[b]];
            var3 += g3[a] * cv * g3[b];
            varxy += gxy[a] * cv * gxy[b];
        }
    }
    if (!(var3 >= 0) || !(varxy >= 0) || !isfinite(var3) || !isfinite(varxy)) return 0;
    *out_sep_se = sqrt(var3);
    *out_sep_xy_se = sqrt(varxy);
    return isfinite(*out_sigma_z_se) && isfinite(*out_sep_se) && isfinite(*out_sep_xy_se);
}


/*
 * 2-spot Levenberg-Marquardt fit with bounds projection. Analytic Jacobian.
 */
int fit_2spot(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    const double *p0,
    const double *lo, const double *hi,
    int max_iter,
    double *out_params,
    double *out_sse,
    int *out_pinned,
    int *out_nfev)
{
    if (n_vox < NPAR + 1) return -1;
    if (max_iter <= 0)    max_iter = MAX_ITER_DEFAULT;

    double p[NPAR];
    memcpy(p, p0, sizeof(p));
    clamp_bounds(p, lo, hi);

    double chi2 = chi2_2spot(p, obs, Z, Y, X, n_vox);
    int    nfev = 1;

    double lambda = 1e-3;
    double JtJ[NPAR * NPAR];
    double Jtr[NPAR];
    double J[NPAR];

    int converged = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        memset(JtJ, 0, sizeof(JtJ));
        memset(Jtr, 0, sizeof(Jtr));

        double A1 = p[0], A2 = p[1];
        double cz1 = p[2], cy1 = p[3], cx1 = p[4];
        double cz2 = p[5], cy2 = p[6], cx2 = p[7];
        double sz  = p[8], sy  = p[9], sx  = p[10], bg = p[11];
        double sz2 = sz*sz, sy2 = sy*sy, sx2 = sx*sx;
        double sz3 = sz*sz2, sy3 = sy*sy2, sx3 = sx*sx2;

        /* Accumulate JtJ and Jtr in one pass. Analytic Jacobian. */
        for (int i = 0; i < n_vox; i++) {
            double z = Z[i], y = Y[i], x = X[i], d = obs[i];
            double dz1 = z - cz1, dy1 = y - cy1, dx1 = x - cx1;
            double dz2 = z - cz2, dy2 = y - cy2, dx2 = x - cx2;
            double e1 = exp(-0.5 * (dz1*dz1/sz2 + dy1*dy1/sy2 + dx1*dx1/sx2));
            double e2 = exp(-0.5 * (dz2*dz2/sz2 + dy2*dy2/sy2 + dx2*dx2/sx2));
            double g1 = A1 * e1, g2 = A2 * e2;
            double f  = g1 + g2 + bg;
            double r  = d - f;

            J[ 0] = e1;                         /* dF/dA1  */
            J[ 1] = e2;                         /* dF/dA2  */
            J[ 2] = g1 * dz1 / sz2;             /* dF/dcz1 */
            J[ 3] = g1 * dy1 / sy2;             /* dF/dcy1 */
            J[ 4] = g1 * dx1 / sx2;             /* dF/dcx1 */
            J[ 5] = g2 * dz2 / sz2;             /* dF/dcz2 */
            J[ 6] = g2 * dy2 / sy2;             /* dF/dcy2 */
            J[ 7] = g2 * dx2 / sx2;             /* dF/dcx2 */
            J[ 8] = (g1 * dz1*dz1 + g2 * dz2*dz2) / sz3;  /* dF/dsz */
            J[ 9] = (g1 * dy1*dy1 + g2 * dy2*dy2) / sy3;  /* dF/dsy */
            J[10] = (g1 * dx1*dx1 + g2 * dx2*dx2) / sx3;  /* dF/dsx */
            J[11] = 1.0;                                   /* dF/dbg */

            for (int a = 0; a < NPAR; a++) {
                Jtr[a] += J[a] * r;
                double Ja = J[a];
                for (int b = 0; b < NPAR; b++)
                    JtJ[a * NPAR + b] += Ja * J[b];
            }
        }

        /* Damped normal equations. */
        double Ad[NPAR * NPAR];
        double bd[NPAR];
        memcpy(Ad, JtJ, sizeof(Ad));
        memcpy(bd, Jtr, sizeof(bd));
        for (int i = 0; i < NPAR; i++)
            Ad[i * NPAR + i] *= (1.0 + lambda);

        double grad_max = 0.0;
        for (int i = 0; i < NPAR; i++) grad_max = fmax(grad_max, fabs(Jtr[i]));
        if (grad_max <= 1e-8 * (1.0 + sqrt(fmax(chi2, 0.0)))) {
            converged = 1;
            break;
        }

        double dp[NPAR];
        if (!solve_linear(Ad, bd, NPAR, dp)) {
            lambda *= 10.0;
            if (lambda > 1e10) break;
            continue;
        }

        double p_new[NPAR];
        for (int i = 0; i < NPAR; i++) {
            p_new[i] = p[i] + dp[i];
        }
        clamp_bounds(p_new, lo, hi);
        double step_max = 0.0;
        for (int i = 0; i < NPAR; i++)
            step_max = fmax(step_max, fabs(p_new[i] - p[i])
                            / (1.0 + fabs(p[i])));

        double chi2_new = chi2_2spot(p_new, obs, Z, Y, X, n_vox);
        nfev++;

        double tol = 1e-12 * (1.0 + fabs(chi2));
        if (isfinite(chi2_new) && chi2_new < chi2) {
            double rel = fabs(chi2 - chi2_new) / (chi2 + 1e-30);
            memcpy(p, p_new, sizeof(p));
            chi2 = chi2_new;
            lambda *= 0.1;
            if (rel < 1e-6 || step_max < 1e-7) { converged = 1; break; }
        } else if (isfinite(chi2_new) && fabs(chi2_new - chi2) <= tol
                   && step_max < 1e-7) {
            converged = 1;
            break;
        } else {
            lambda *= 10.0;
            if (lambda > 1e10) break;
        }
    }

    /* Non-finite check. */
    for (int i = 0; i < NPAR; i++) {
        if (!isfinite(p[i])) {
            memcpy(out_params, p, sizeof(p));
            *out_sse    = chi2;
            *out_pinned = 0;
            *out_nfev   = nfev;
            return 2;
        }
    }

    /* "Pinned to a boundary" flag: any finite-bounded param within 1%
     * of either endpoint. Matches the Python implementation. */
    int pinned = 0;
    for (int i = 0; i < NPAR; i++) {
        if (isfinite(lo[i]) && isfinite(hi[i]) && (hi[i] - lo[i]) > 0) {
            double eps = 0.01 * (hi[i] - lo[i]);
            if (p[i] - lo[i] < eps || hi[i] - p[i] < eps) { pinned = 1; break; }
        }
    }

    memcpy(out_params, p, sizeof(p));
    *out_sse    = chi2;
    *out_pinned = pinned;
    *out_nfev   = nfev;
    return converged ? 0 : 1;
}


/*
 * ============================================================================
 * batch_replication_qc  - full-TP entry point, OpenMP-parallel over cells.
 * ============================================================================
 *
 * One call handles every cell of a single TP image. Each cell is independent,
 * so the outer loop is parallelized with OpenMP. Per-thread scratch buffers
 * sized to the largest bbox are allocated once inside the parallel region.
 *
 * Batch semantics:
 *   - Skip cells without a 1-spot fit (status = 0, row not written by Python)
 *   - Skip cells with n_vox < min_n_voxels (status = 1, also skipped by Python)
 *   - Refit M1 on the owning mask plus physical background shell
 *   - Run Stage A diagnostics (never used as a fit screen)
 *   - Run four-start M2 and apply the uncertainty-aware replication call
 *       * singular / non-finite / invalid -> status in {5, 6, 7}
 *       * tested_not_confirmed -> status = 3
 *       * confirmed -> status = 4
 *
 * Outputs are two contiguous arrays, one row per cell:
 *
 *   out_floats[c * N_COLS_FLOAT + k]  (N_COLS_FLOAT = 41)
 *     0 BIC_1
 *     1 BIC_2            (NaN if Stage B skipped or failed)
 *     2 delta_BIC        (NaN if Stage B skipped or failed)
 *     3 spot_sep_nm      (NaN if Stage B skipped or failed)
 *     4 amp1
 *     5 amp2             (NaN if Stage B not run)
 *     6 sigma_x
 *     7 sigma_y
 *     8 sigma_z
 *     9 resid_score
 *    10 peak_rel_amp
 *    11 peak_offset_px
 *    12 sse1             (1-spot SSE, always populated if Stage A ran)
 *    13 sse2             (2-spot SSE, NaN if Stage B skipped or failed)
 *    14 amp_recovery     (max(A1,A2)/amp_1spot; NaN if Stage B skipped or failed)
 *    15 peak_support_c1  (raw at fitted c1 vs raw max in small ball; ~1 = real peak)
 *    16 peak_support_c2  (same for fitted c2; <1 = on a slope/shoulder)
 *    17 valley_frac      (min raw along c1<->c2 line / min raw at the centers,
 *                         all bg-subtracted; <1 = real valley, >=1 = shoulder)
 *    18 resid_core_frac  (positive 1-spot residual mass inside core ball at Stage A
 *                         peak / total positive 1-spot residual; high = compact lobe,
 *                         low = diffuse smear)
 *    19 spot_sep_xy_nm   (lateral-only fitted spot separation sqrt(dy^2+dx^2) in nm;
 *                         NaN if Stage B skipped or failed. The z component of
 *                         spot_sep_nm is unreliable on thin stacks where the axial
 *                         fit is unidentifiable — gate on this for a robust call.)
 *    20..23 refitted M1 amplitude, sigma_z, sigma_y, sigma_x
 *    24..29 fitted M2 centers c1(z,y,x), c2(z,y,x)
 *    30 sigma_z_se; 31..34 separation SE/95% LCB for 3D and XY
 *    35 effective sigma_z upper guard
 *    36..39 refitted M1 center(z,y,x), background; 40 M2 background
 *
 *   out_ints[c * N_COLS_INT + k]  (N_COLS_INT = 24)
 *     0 status     (as above)
 *     1 n_voxels
 *     2 replication_suspect   (0/1)
 *     3 replication_confirmed (0/1)
 *     4 n_spots_best          (1 or 2)
 *     5 sigma_xy_broad        (0/1)
 *     6 sigma_z_broad         (0/1)  [diagnostic only; NOT a suspect trigger]
 *     7 resid_broad           (0/1)
 *     8 two_lobe_suspect      (0/1)
 *     9 fit2_pinned           (0/1)  [any 2-spot param on a bbox/sigma bound;
 *                                     OR of fit2_center_pinned and fit2_sigma_pinned;
 *                                     kept for backward compatibility]
 *    10 fit2_center_pinned    (0/1)  [any of the 6 center coords (cz1,cy1,cx1,
 *                                     cz2,cy2,cx2) sits within 1% of its bbox
 *                                     bound — used by downstream merged-mask QC]
 *    11 fit2_sigma_pinned     (0/1)  [any of sigma_z, sigma_y, sigma_x sits
 *                                     within 1% of its sigma bound]
 *    12 M1 return code; 13 M2 return code; 14 M2 nfev; 15 M2 starts
 *    16..17 rounded mask labels under the two centers
 *    18 sigma_z upper-censored; 19 axial-identifiable
 *    20..21 center-in-owned-support flags; 22 lower-width-pinned;
 *    23 sigma_xy upper-pinned
 */

#define NCOLS_FLOAT 41
#define NCOLS_INT   24

/* Default geometry of the small neighbourhoods used by the new raw-peak /
 * compactness metrics. XY radius is in voxel units (~108 nm/vx); Z radius
 * in slice units (~100 nm/slice). Tuned roughly to one PSF e^-1 width so
 * the ball captures one spot but not a neighbouring spot when the two
 * fitted centres are at the production min_spot_sep_nm = 500 nm separation.
 *   500 nm in XY ≈ 4.6 vx; ball radius 1.5 vx is well below.
 *   500 nm in Z  = 5 slices; ball radius 1.0 slice is well below.
 */
#define PEAK_BALL_R_XY  1.5
#define PEAK_BALL_R_Z   1.0
#define RESID_CORE_R_XY 1.5
#define RESID_CORE_R_Z  1.0
#define VALLEY_N_SAMPLES 5    /* number of points along c1<->c2 (excl. endpoints) */

#define STATUS_NOT_PRESENT          0
#define STATUS_TOO_FEW_VOXELS       1
#define STATUS_NOT_SUSPECT          2
#define STATUS_TESTED_NOT_CONFIRMED 3
#define STATUS_CONFIRMED            4
#define STATUS_2SPOT_FAIL_NONCONV   5  /* fit_2spot returned 1 (max_iter) */
#define STATUS_2SPOT_FAIL_NONFINITE 6  /* fit_2spot returned 2             */
#define STATUS_2SPOT_FAIL_INVALID   7  /* fit_2spot returned -1 (bad input)*/
#define STATUS_1SPOT_FAIL_NONCONV   8
#define STATUS_1SPOT_FAIL_INVALID   9

static inline double nan_d(void) { return NAN; }

static int in_owned_support(const int *labels, int nz, int ny, int nx,
                            int cell_label, int iz, int iy, int ix,
                            double shell_nm, double z_step_nm, double xy_pixel_nm)
{
    if (iz < 0 || iz >= nz || iy < 0 || iy >= ny || ix < 0 || ix >= nx) return 0;
    int idx = iz*ny*nx + iy*nx + ix;
    int lv = labels[idx];
    if (lv == cell_label) return 1;
    if (lv != 0 || shell_nm <= 0) return 0;
    int rz = (int)ceil(shell_nm / z_step_nm);
    int rxy = (int)ceil(shell_nm / xy_pixel_nm);
    double shell2 = shell_nm * shell_nm;
    for (int dz = -rz; dz <= rz; dz++) {
        int zz = iz + dz;
        if (zz < 0 || zz >= nz) continue;
        for (int dy = -rxy; dy <= rxy; dy++) {
            int yy = iy + dy;
            if (yy < 0 || yy >= ny) continue;
            for (int dx = -rxy; dx <= rxy; dx++) {
                int xx = ix + dx;
                if (xx < 0 || xx >= nx) continue;
                double d2 = sqr(dz*z_step_nm) + sqr(dy*xy_pixel_nm)
                          + sqr(dx*xy_pixel_nm);
                if (d2 <= shell2
                    && labels[zz*ny*nx + yy*nx + xx] == cell_label) return 1;
            }
        }
    }
    return 0;
}


/*
 * Look up the raw image value at the integer voxel nearest (cz, cy, cx),
 * provided that voxel is inside the cell's bbox AND inside the cell mask.
 * Returns NAN if the rounded voxel is out of bounds or outside the mask.
 *
 * Used by the new raw-peak support metrics. The "nearest voxel" choice
 * mirrors how the localizer reports centers (sub-voxel) but verifies them
 * against an integer-grid raw signal.
 */
static double raw_at_nearest(const double *image, const int *mask_labels,
                             int nz, int ny, int nx, int cell_label,
                             int bz0, int bz1, int by0, int by1, int bx0, int bx1,
                             double cz, double cy, double cx,
                             double shell_nm, double z_step_nm, double xy_pixel_nm)
{
    int iz = (int)round(cz), iy = (int)round(cy), ix = (int)round(cx);
    if (iz < bz0 || iz > bz1 || iy < by0 || iy > by1 || ix < bx0 || ix > bx1)
        return NAN;
    int idx = iz * ny * nx + iy * nx + ix;
    if (!in_owned_support(mask_labels, nz, ny, nx, cell_label, iz, iy, ix,
                          shell_nm, z_step_nm, xy_pixel_nm)) return NAN;
    return image[idx];
}


/*
 * Find the maximum raw image value within an anisotropic ball of half-axes
 * (r_z in slices, r_xy in pixels) around (cz, cy, cx). The ball is
 * intersected with the cell's bbox and the cell mask. Returns NAN if no
 * voxel is found.
 *
 * Used to compute peak_support = (raw_at_center - bg) / (raw_max_in_ball - bg).
 * peak_support ~ 1 means the center sits on a real local maximum;
 * peak_support << 1 means the center sits on a slope/shoulder.
 */
static double raw_max_in_ball(const double *image, const int *mask_labels,
                              int nz, int ny, int nx, int cell_label,
                              int bz0, int bz1, int by0, int by1, int bx0, int bx1,
                              double cz, double cy, double cx,
                              double r_z, double r_xy,
                              double shell_nm, double z_step_nm, double xy_pixel_nm)
{
    int iz_c = (int)round(cz), iy_c = (int)round(cy), ix_c = (int)round(cx);
    int rz  = (int)ceil(r_z);
    int rxy = (int)ceil(r_xy);
    double maxv = -INFINITY;
    for (int dz = -rz; dz <= rz; dz++) {
        int iz = iz_c + dz;
        if (iz < bz0 || iz > bz1) continue;
        for (int dy = -rxy; dy <= rxy; dy++) {
            int iy = iy_c + dy;
            if (iy < by0 || iy > by1) continue;
            for (int dx = -rxy; dx <= rxy; dx++) {
                int ix = ix_c + dx;
                if (ix < bx0 || ix > bx1) continue;
                /* Anisotropic ball test in voxel-coord units. */
                double rrz = ((double)dz) / r_z;
                double rry = ((double)dy) / r_xy;
                double rrx = ((double)dx) / r_xy;
                if (rrz*rrz + rry*rry + rrx*rrx > 1.0) continue;
                int idx = iz * ny * nx + iy * nx + ix;
                if (!in_owned_support(mask_labels, nz, ny, nx, cell_label,
                                      iz, iy, ix, shell_nm,
                                      z_step_nm, xy_pixel_nm)) continue;
                double v = image[idx];
                if (v > maxv) maxv = v;
            }
        }
    }
    return isfinite(maxv) ? maxv : NAN;
}


/*
 * Sample raw image at K interior points along the line from c1 to c2 and
 * return the minimum (NaN if any sample is outside the mask).
 *
 *   valley_frac = (min_along_line - bg) / (min(raw_at_c1, raw_at_c2) - bg)
 *
 * Real compact double: a saddle exists, so valley_frac < 1 (often ~0.3-0.7).
 * Shoulder/edge FP: monotonic intensity, valley_frac >= 1.
 */
static double raw_min_along_line(const double *image, const int *mask_labels,
                                 int nz, int ny, int nx, int cell_label,
                                 int bz0, int bz1, int by0, int by1, int bx0, int bx1,
                                 double c1z, double c1y, double c1x,
                                 double c2z, double c2y, double c2x,
                                 int n_samples,
                                 double shell_nm, double z_step_nm, double xy_pixel_nm)
{
    if (n_samples < 1) return NAN;
    double minv = INFINITY;
    int valid = 0;
    for (int k = 1; k <= n_samples; k++) {
        double t = (double)k / (double)(n_samples + 1);
        double cz = c1z + t * (c2z - c1z);
        double cy = c1y + t * (c2y - c1y);
        double cx = c1x + t * (c2x - c1x);
        double v = raw_at_nearest(image, mask_labels, nz, ny, nx, cell_label,
                                  bz0, bz1, by0, by1, bx0, bx1, cz, cy, cx,
                                  shell_nm, z_step_nm, xy_pixel_nm);
        if (!isfinite(v)) continue;
        if (v < minv) minv = v;
        valid++;
    }
    return valid > 0 ? minv : NAN;
}


/*
 * Concentration of the positive 1-spot residual around the Stage A
 * residual peak. Walks the same masked voxel list the Stage A pass used.
 *
 *   resid_core_frac = sum(positive 1-spot residual inside core ball) /
 *                     sum(total positive 1-spot residual)
 *
 * High = compact lobe; low = diffuse shoulder smear. Independent of the
 * Stage B fit — characterises the suspect *feature*, not the fitted model.
 */
static double resid_core_concentration(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox,
    double amp1, double cz1, double cy1, double cx1,
    double sz1, double sy1, double sx1, double bg1,
    double peak_z, double peak_y, double peak_x,
    double r_z, double r_xy)
{
    double total_pos = 0.0;
    double core_pos  = 0.0;
    double r_z_inv  = 1.0 / r_z;
    double r_xy_inv = 1.0 / r_xy;
    for (int i = 0; i < n_vox; i++) {
        double g = gauss3d(amp1, cz1, cy1, cx1, sz1, sy1, sx1,
                           Z[i], Y[i], X[i]);
        double r = obs[i] - (g + bg1);
        if (r <= 0) continue;
        total_pos += r;
        double dz = (Z[i] - peak_z) * r_z_inv;
        double dy = (Y[i] - peak_y) * r_xy_inv;
        double dx = (X[i] - peak_x) * r_xy_inv;
        if (dz*dz + dy*dy + dx*dx <= 1.0) core_pos += r;
    }
    if (total_pos <= 0) return NAN;
    return core_pos / total_pos;
}

static void residual_seed_centers(
    const double *obs, const double *Z, const double *Y, const double *X,
    int n_vox, const double *p1, double seeds[2][3])
{
    double best[2] = {-INFINITY, -INFINITY};
    int best_i[2] = {-1, -1};
    for (int i = 0; i < n_vox; i++) {
        double model1 = gauss3d(p1[0], p1[1], p1[2], p1[3],
                                p1[4], p1[5], p1[6], Z[i], Y[i], X[i]) + p1[7];
        double r = obs[i] - model1;
        double dc = sqrt(sqr(Z[i]-p1[1]) + sqr(Y[i]-p1[2]) + sqr(X[i]-p1[3]));
        if (dc < 1.0) continue;
        if (r > best[0]) { best[0] = r; best_i[0] = i; }
    }
    for (int i = 0; i < n_vox; i++) {
        if (best_i[0] >= 0) {
            double d = sqrt(sqr(Z[i]-Z[best_i[0]]) + sqr(Y[i]-Y[best_i[0]])
                          + sqr(X[i]-X[best_i[0]]));
            if (d < 1.0) continue;
        }
        double dc = sqrt(sqr(Z[i]-p1[1]) + sqr(Y[i]-p1[2]) + sqr(X[i]-p1[3]));
        if (dc < 1.0) continue;
        double model1 = gauss3d(p1[0], p1[1], p1[2], p1[3],
                                p1[4], p1[5], p1[6], Z[i], Y[i], X[i]) + p1[7];
        double r = obs[i] - model1;
        if (r > best[1]) { best[1] = r; best_i[1] = i; }
    }
    for (int s = 0; s < 2; s++) {
        int i = best_i[s];
        if (i < 0) {
            double far = -1.0;
            for (int j = 0; j < n_vox; j++) {
                double d = sqr(Z[j]-p1[1]) + sqr(Y[j]-p1[2]) + sqr(X[j]-p1[3]);
                if (d > far) { far = d; i = j; }
            }
        }
        seeds[s][0] = Z[i]; seeds[s][1] = Y[i]; seeds[s][2] = X[i];
    }
}


void batch_replication_qc(
    const double *image,
    const int    *mask_labels,
    int nz, int ny, int nx,

    const int    *bboxes,     /* [n_cells * 6] */
    const int    *has_fit1,   /* [n_cells]     */
    const double *fit1,       /* [n_cells * 8] amp,cz,cy,cx,sz,sy,sx,bg */
    int n_cells,

    double sigma_xy_broad_thr,
    double sigma_z_broad_thr,
    double resid_score_thr,
    double min_lobe_sep_px,
    double min_lobe_rel_amp,
    double delta_bic_thr,
    double min_spot_sep_nm,
    double min_amp_ratio,
    double min_amp_recovery,
    /* New raw-peak / compactness gates. Set to 0 (or, for the max gate,
     * any non-positive value) to disable. When disabled the metrics are
     * still computed and emitted to the side-car CSV. */
    double min_peak_support_c2,   /* 0 disables */
    double max_valley_frac,       /* <=0 disables */
    double min_resid_core_frac,   /* 0 disables */

    double sigma_z_max,           /* <=0: crop-depth numerical guard only */
    double center_shell_nm,

    double z_step_nm,
    double xy_pixel_nm,

    int min_n_voxels,
    int max_iter_lm,

    double *out_floats,    /* [n_cells * NCOLS_FLOAT] */
    int    *out_ints       /* [n_cells * NCOLS_INT]   */)
{
    /* Determine the largest crop so each thread can allocate scratch once. */
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
    /* Initialize every output row to "skip". Threads will overwrite as they
     * assign work; rows for cells without fit1 are left at status=0. */
    for (int c = 0; c < n_cells; c++) {
        int *ri = &out_ints[c * NCOLS_INT];
        double *rf = &out_floats[c * NCOLS_FLOAT];
        for (int k = 0; k < NCOLS_INT; k++)   ri[k] = 0;
        for (int k = 0; k < NCOLS_FLOAT; k++) rf[k] = nan_d();
        ri[0] = STATUS_NOT_PRESENT;
    }
    if (max_cn <= 0) return;

    #pragma omp parallel
    {
        /* Thread-local scratch buffers. */
        double *obs = (double *)malloc((size_t)max_cn * sizeof(double));
        double *Zb  = (double *)malloc((size_t)max_cn * sizeof(double));
        double *Yb  = (double *)malloc((size_t)max_cn * sizeof(double));
        double *Xb  = (double *)malloc((size_t)max_cn * sizeof(double));
        int alloc_ok = (obs && Zb && Yb && Xb);

        /* Null-calibration batches mix crop sizes and can be smaller than the
         * previous chunk size of 16. One-row dynamic scheduling keeps all
         * allocated cores busy without changing any fit semantics. */
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_cells; c++) {
            int *ri = &out_ints[c * NCOLS_INT];
            double *rf = &out_floats[c * NCOLS_FLOAT];

            if (!alloc_ok) { ri[0] = STATUS_2SPOT_FAIL_INVALID; continue; }
            if (!has_fit1[c]) continue;   /* stays at STATUS_NOT_PRESENT */

            int bz0 = bboxes[c*6+0], bz1 = bboxes[c*6+1];
            int by0 = bboxes[c*6+2], by1 = bboxes[c*6+3];
            int bx0 = bboxes[c*6+4], bx1 = bboxes[c*6+5];

            /* Fit support = owning mask plus a physical one-voxel shell.
             * Background voxels in another labelled cell are never included. */
            int n_vox = 0;
            for (int iz = bz0; iz <= bz1; iz++)
                for (int iy = by0; iy <= by1; iy++)
                    for (int ix = bx0; ix <= bx1; ix++) {
                        int idx = iz * ny * nx + iy * nx + ix;
                        if (in_owned_support(mask_labels, nz, ny, nx, c + 1,
                                             iz, iy, ix, center_shell_nm,
                                             z_step_nm, xy_pixel_nm)) {
                            obs[n_vox] = image[idx];
                            Zb[n_vox]  = (double)iz;
                            Yb[n_vox]  = (double)iy;
                            Xb[n_vox]  = (double)ix;
                            n_vox++;
                        }
                    }

            if (n_vox < min_n_voxels) {
                ri[0] = STATUS_TOO_FEW_VOXELS;
                ri[1] = n_vox;
                continue;
            }
            ri[1] = n_vox;

            /* Refit M1 on exactly the same support and bounds used by M2. */
            const double *f1 = &fit1[c * 8];
            int cnz = bz1 - bz0 + 1, cny = by1 - by0 + 1, cnx = bx1 - bx0 + 1;
            const double MIN_SZ = 0.2, MIN_SXY = 0.3;
            double max_sz = (sigma_z_max > 0.0) ? fmin(sigma_z_max, (double)cnz)
                                                 : (double)cnz;
            double max_sy = fmax(MIN_SXY, cny / 2.0);
            double max_sx = fmax(MIN_SXY, cnx / 2.0);
            max_sz = fmax(MIN_SZ, max_sz);

            double p1_seed[NPAR1] = {
                fmax(0.0, f1[0]), f1[1], f1[2], f1[3],
                fmin(max_sz, fmax(MIN_SZ, f1[4])),
                fmin(max_sy, fmax(MIN_SXY, f1[5])),
                fmin(max_sx, fmax(MIN_SXY, f1[6])),
                fmax(0.0, f1[7]),
            };
            if (!center_supported(p1_seed[1], p1_seed[2], p1_seed[3],
                                  Zb, Yb, Xb, n_vox)) {
                int imax = 0;
                for (int i = 1; i < n_vox; i++) if (obs[i] > obs[imax]) imax = i;
                p1_seed[1] = Zb[imax]; p1_seed[2] = Yb[imax]; p1_seed[3] = Xb[imax];
            }
            double lo1[NPAR1] = {0.0, bz0, by0, bx0, MIN_SZ, MIN_SXY, MIN_SXY, 0.0};
            double hi1[NPAR1] = {INFINITY, bz1, by1, bx1, max_sz, max_sy, max_sx, INFINITY};
            double params1[NPAR1], sse1 = 0.0;
            int nfev1 = 0;
            int rc1 = fit_1spot(obs, Zb, Yb, Xb, n_vox, p1_seed, lo1, hi1,
                                max_iter_lm, params1, &sse1, &nfev1);
            ri[12] = rc1;
            if (rc1 == -1 || rc1 == 2) { ri[0] = STATUS_1SPOT_FAIL_INVALID; continue; }
            if (rc1 != 0) { ri[0] = STATUS_1SPOT_FAIL_NONCONV; continue; }

            double amp1 = params1[0], cz1 = params1[1], cy1 = params1[2], cx1 = params1[3];
            double sz1 = params1[4], sy1 = params1[5], sx1 = params1[6], bg1 = params1[7];

            /* Stage A is diagnostic only; both models are evaluated for every row. */
            double resid_score = 0, peak_rel_amp = 0, peak_off_px = 0;
            double peak_z = cz1, peak_y = cy1, peak_x = cx1;
            stage_a_and_sse1(obs, Zb, Yb, Xb, n_vox,
                             amp1, cz1, cy1, cx1, sz1, sy1, sx1, bg1,
                             &resid_score, &peak_rel_amp, &peak_off_px,
                             &peak_z, &peak_y, &peak_x, &sse1);

            double bic1 = (isfinite(sse1) && n_vox > 0)
                          ? n_vox * log(fmax(sse1, 1e-24) / n_vox)
                            + 8.0 * log((double)n_vox)
                          : INFINITY;

            int sigma_xy_broad = (fmax(sx1, sy1) > sigma_xy_broad_thr);
            int sigma_z_broad  = (sz1 > sigma_z_broad_thr);
            int resid_broad    = (resid_score > resid_score_thr);
            int two_lobe       = (peak_off_px >= min_lobe_sep_px &&
                                  peak_rel_amp >= min_lobe_rel_amp);
            /* sigma_z_broad is REPORTED but NOT a suspect trigger. On the
             * genome WF data it fires on nearly every cell (PSF is
             * physically broad in z), which was drowning Stage B in work
             * and inflating FPs. Dropped from the OR; kept as a column. */
            int suspect = (sigma_xy_broad || resid_broad || two_lobe);

            /* Common fields. */
            rf[0]  = bic1;
            rf[4]  = amp1;
            rf[6]  = sx1;
            rf[7]  = sy1;
            rf[8]  = sz1;
            rf[9]  = resid_score;
            rf[10] = peak_rel_amp;
            rf[11] = peak_off_px;
            rf[12] = sse1;
            rf[20] = amp1;
            rf[21] = sz1;
            rf[22] = sy1;
            rf[23] = sx1;
            rf[35] = max_sz;
            rf[36] = cz1;
            rf[37] = cy1;
            rf[38] = cx1;
            rf[39] = bg1;
            ri[2]  = suspect ? 1 : 0;
            ri[4]  = 1;
            ri[5]  = sigma_xy_broad;
            ri[6]  = sigma_z_broad;
            ri[7]  = resid_broad;
            ri[8]  = two_lobe;

            /* M2 uses the same nuisance-parameter bounds as M1. */
            double sz_init = fmin(max_sz, fmax(MIN_SZ, sz1));
            double sy_init = fmin(max_sy, fmax(MIN_SXY, sy1));
            double sx_init = fmin(max_sx, fmax(MIN_SXY, sx1));

            double lo[NPAR] = {
                0.0, 0.0,
                (double)bz0, (double)by0, (double)bx0,
                (double)bz0, (double)by0, (double)bx0,
                MIN_SZ, MIN_SXY, MIN_SXY,
                0.0,
            };
            double hi[NPAR] = {
                INFINITY, INFINITY,
                (double)bz1, (double)by1, (double)bx1,
                (double)bz1, (double)by1, (double)bx1,
                max_sz, max_sy, max_sx,
                INFINITY,
            };

            double seeds[2][3];
            residual_seed_centers(obs, Zb, Yb, Xb, n_vox, params1, seeds);
            const double split[2] = {0.3, 0.5};
            double params2[NPAR], sse2 = INFINITY;
            int pinned = 0, nfev = 0, rc = 1, n_starts = 0;
            for (int si = 0; si < 2; si++) {
                for (int ai = 0; ai < 2; ai++) {
                    double f2 = split[ai];
                    double p0[NPAR] = {
                        (1.0-f2)*amp1, f2*amp1,
                        cz1, cy1, cx1,
                        seeds[si][0], seeds[si][1], seeds[si][2],
                        sz_init, sy_init, sx_init, bg1,
                    };
                    double trial[NPAR], trial_sse = INFINITY;
                    int trial_pinned = 0, trial_nfev = 0;
                    int trial_rc = fit_2spot(obs, Zb, Yb, Xb, n_vox,
                                             p0, lo, hi, max_iter_lm,
                                             trial, &trial_sse,
                                             &trial_pinned, &trial_nfev);
                    n_starts++;
                    if (trial_rc == 0 && isfinite(trial_sse) && trial_sse < sse2) {
                        memcpy(params2, trial, sizeof(params2));
                        sse2 = trial_sse;
                        pinned = trial_pinned;
                        nfev = trial_nfev;
                        rc = 0;
                    }
                }
            }
            ri[13] = rc;
            ri[14] = nfev;
            ri[15] = n_starts;
            if (rc != 0) { ri[0] = STATUS_2SPOT_FAIL_NONCONV; continue; }

            double bic2 = (isfinite(sse2) && n_vox > 0)
                          ? n_vox * log(fmax(sse2, 1e-24) / n_vox)
                            + 12.0 * log((double)n_vox)
                          : INFINITY;
            double delta_bic = bic1 - bic2;

            double A1 = params2[0], A2 = params2[1];
            double cz1_fit = params2[2], cy1_fit = params2[3], cx1_fit = params2[4];
            double cz2_fit = params2[5], cy2_fit = params2[6], cx2_fit = params2[7];
            double dz_nm = (cz1_fit - cz2_fit) * z_step_nm;
            double dy_nm = (cy1_fit - cy2_fit) * xy_pixel_nm;
            double dx_nm = (cx1_fit - cx2_fit) * xy_pixel_nm;
            double sep_nm = sqrt(dz_nm*dz_nm + dy_nm*dy_nm + dx_nm*dx_nm);
            double sep_xy_nm = sqrt(dy_nm*dy_nm + dx_nm*dx_nm);

            double amp_max = fmax(A1, A2);
            double amp_ratio = (amp_max > 0) ? fmin(A1, A2) / amp_max : 0.0;
            double amp_recovery = (amp1 > 0) ? amp_max / amp1 : 0.0;

            /* New raw-peak / compactness diagnostics. Computed unconditionally
             * after Stage B succeeds; gated only if the corresponding CLI
             * thresholds are active. The brighter fitted spot is labelled c1
             * for the per-center metric so that peak_support_c1 is "the
             * stronger spot's anchoring on a real peak" (typically the real
             * locus); peak_support_c2 is the weaker spot, which is where
             * shoulder-split FPs fail.
             */
            double cA_z, cA_y, cA_x, cB_z, cB_y, cB_x;
            if (A1 >= A2) {
                cA_z = cz1_fit; cA_y = cy1_fit; cA_x = cx1_fit;
                cB_z = cz2_fit; cB_y = cy2_fit; cB_x = cx2_fit;
            } else {
                cA_z = cz2_fit; cA_y = cy2_fit; cA_x = cx2_fit;
                cB_z = cz1_fit; cB_y = cy1_fit; cB_x = cx1_fit;
            }
            double bg2 = params2[11];
            double rawA = raw_at_nearest(image, mask_labels, nz, ny, nx, c + 1,
                                         bz0, bz1, by0, by1, bx0, bx1,
                                         cA_z, cA_y, cA_x,
                                         center_shell_nm, z_step_nm, xy_pixel_nm);
            double rawB = raw_at_nearest(image, mask_labels, nz, ny, nx, c + 1,
                                         bz0, bz1, by0, by1, bx0, bx1,
                                         cB_z, cB_y, cB_x,
                                         center_shell_nm, z_step_nm, xy_pixel_nm);
            double maxA = raw_max_in_ball(image, mask_labels, nz, ny, nx, c + 1,
                                          bz0, bz1, by0, by1, bx0, bx1,
                                          cA_z, cA_y, cA_x,
                                          PEAK_BALL_R_Z, PEAK_BALL_R_XY,
                                          center_shell_nm, z_step_nm, xy_pixel_nm);
            double maxB = raw_max_in_ball(image, mask_labels, nz, ny, nx, c + 1,
                                          bz0, bz1, by0, by1, bx0, bx1,
                                          cB_z, cB_y, cB_x,
                                          PEAK_BALL_R_Z, PEAK_BALL_R_XY,
                                          center_shell_nm, z_step_nm, xy_pixel_nm);
            double peak_support_c1 = (isfinite(rawA) && isfinite(maxA)
                                      && (maxA - bg2) > 0)
                                     ? (rawA - bg2) / (maxA - bg2) : NAN;
            double peak_support_c2 = (isfinite(rawB) && isfinite(maxB)
                                      && (maxB - bg2) > 0)
                                     ? (rawB - bg2) / (maxB - bg2) : NAN;

            double valley_min = raw_min_along_line(
                image, mask_labels, nz, ny, nx, c + 1,
                bz0, bz1, by0, by1, bx0, bx1,
                cA_z, cA_y, cA_x, cB_z, cB_y, cB_x, VALLEY_N_SAMPLES,
                center_shell_nm, z_step_nm, xy_pixel_nm);
            double anchor = (isfinite(rawA) && isfinite(rawB))
                            ? fmin(rawA, rawB) : NAN;
            double valley_frac = (isfinite(valley_min) && isfinite(anchor)
                                  && (anchor - bg2) > 0)
                                 ? (valley_min - bg2) / (anchor - bg2) : NAN;

            double resid_core_frac = resid_core_concentration(
                obs, Zb, Yb, Xb, n_vox,
                amp1, cz1, cy1, cx1, sz1, sy1, sx1, bg1,
                peak_z, peak_y, peak_x,
                RESID_CORE_R_Z, RESID_CORE_R_XY);

            /* Optional Stage B gates. Every gate added with the new
             * diagnostic metrics is "disabled when its threshold is
             * non-positive" — including amp_recovery, which used to be on
             * by default. Caller defaults in replication_qc.py keep all of
             * these disabled out of the box; the orchestrator opts in
             * explicitly to preserve production behaviour. NaN metrics are
             * treated as failing only when the corresponding gate is
             * active. */
            int amp_recovery_ok = (min_amp_recovery   <= 0.0)
                                  || (isfinite(amp_recovery)
                                      && amp_recovery   >= min_amp_recovery);
            int peak_support_ok = (min_peak_support_c2 <= 0.0)
                                  || (isfinite(peak_support_c2)
                                      && peak_support_c2 >= min_peak_support_c2);
            int valley_ok       = (max_valley_frac    <= 0.0)
                                  || (isfinite(valley_frac)
                                      && valley_frac    <= max_valley_frac);
            int core_ok         = (min_resid_core_frac <= 0.0)
                                  || (isfinite(resid_core_frac)
                                      && resid_core_frac >= min_resid_core_frac);

            int iz1 = (int)lround(cz1_fit), iy1 = (int)lround(cy1_fit), ix1 = (int)lround(cx1_fit);
            int iz2 = (int)lround(cz2_fit), iy2 = (int)lround(cy2_fit), ix2 = (int)lround(cx2_fit);
            int center1_label = mask_labels[iz1*ny*nx + iy1*nx + ix1];
            int center2_label = mask_labels[iz2*ny*nx + iy2*nx + ix2];
            int center1_supported = in_owned_support(mask_labels, nz, ny, nx, c+1,
                                                     iz1, iy1, ix1, center_shell_nm,
                                                     z_step_nm, xy_pixel_nm);
            int center2_supported = in_owned_support(mask_labels, nz, ny, nx, c+1,
                                                     iz2, iy2, ix2, center_shell_nm,
                                                     z_step_nm, xy_pixel_nm);

            int center_pinned = 0, sigma_pinned = 0, lower_width_pinned = 0;
            int xy_upper_pinned = 0;
            for (int i = 2; i <= 7; i++) {
                double eps = 0.01 * (hi[i] - lo[i]);
                if (params2[i] - lo[i] < eps || hi[i] - params2[i] < eps) {
                    center_pinned = 1;
                    break;
                }
            }
            for (int i = 8; i <= 10; i++) {
                double eps = 0.01 * (hi[i] - lo[i]);
                if (params2[i] - lo[i] < eps) lower_width_pinned = 1;
                if (i >= 9 && hi[i] - params2[i] < eps) xy_upper_pinned = 1;
                if (params2[i] - lo[i] < eps || hi[i] - params2[i] < eps)
                    sigma_pinned = 1;
            }
            double z_eps = 0.01 * (hi[8] - lo[8]);
            int sigma_z_censored = (hi[8] - params2[8] < z_eps);

            double sigma_z_se = NAN, sep_se = NAN, sep_xy_se = NAN;
            int uncertainty_ok = fit2_uncertainty(
                Zb, Yb, Xb, n_vox, params2, sse2,
                z_step_nm, xy_pixel_nm, &sigma_z_se, &sep_se, &sep_xy_se);
            double sep_lcb = uncertainty_ok ? fmax(0.0, sep_nm - 1.96*sep_se) : NAN;
            double sep_xy_lcb = uncertainty_ok ? fmax(0.0, sep_xy_nm - 1.96*sep_xy_se) : NAN;
            int axial_identifiable = uncertainty_ok && !sigma_z_censored
                                  && sigma_z_se <= params2[8];
            double separation_lcb = axial_identifiable ? sep_lcb : sep_xy_lcb;

            int confirmed = (delta_bic >= delta_bic_thr
                          && isfinite(separation_lcb)
                          && separation_lcb >= min_spot_sep_nm
                          && amp_ratio >= min_amp_ratio
                          && !center_pinned
                          && !lower_width_pinned
                          && !xy_upper_pinned
                          && center1_supported && center2_supported
                          && A1 > 0 && A2 > 0
                          && amp_recovery_ok
                          && peak_support_ok
                          && valley_ok
                          && core_ok) ? 1 : 0;

            /* Overwrite sigmas + amps with the 2-spot values. */
            rf[1]  = bic2;
            rf[2]  = delta_bic;
            rf[3]  = sep_nm;
            rf[4]  = A1;
            rf[5]  = A2;
            rf[6]  = params2[10];  /* sigma_x */
            rf[7]  = params2[9];   /* sigma_y */
            rf[8]  = params2[8];   /* sigma_z */
            rf[13] = sse2;
            rf[14] = amp_recovery;
            rf[15] = peak_support_c1;
            rf[16] = peak_support_c2;
            rf[17] = valley_frac;
            rf[18] = resid_core_frac;
            rf[19] = sep_xy_nm;
            rf[24] = cz1_fit; rf[25] = cy1_fit; rf[26] = cx1_fit;
            rf[27] = cz2_fit; rf[28] = cy2_fit; rf[29] = cx2_fit;
            rf[30] = sigma_z_se;
            rf[31] = sep_se;
            rf[32] = sep_lcb;
            rf[33] = sep_xy_se;
            rf[34] = sep_xy_lcb;
            rf[40] = bg2;

            ri[3]  = confirmed;
            ri[4]  = confirmed ? 2 : 1;
            ri[9]  = pinned;
            ri[10] = center_pinned;
            ri[11] = sigma_pinned;
            ri[16] = center1_label;
            ri[17] = center2_label;
            ri[18] = sigma_z_censored;
            ri[19] = axial_identifiable;
            ri[20] = center1_supported;
            ri[21] = center2_supported;
            ri[22] = lower_width_pinned;
            ri[23] = xy_upper_pinned;
            ri[0]  = confirmed ? STATUS_CONFIRMED : STATUS_TESTED_NOT_CONFIRMED;
        }

        free(obs); free(Zb); free(Yb); free(Xb);
    }
}
