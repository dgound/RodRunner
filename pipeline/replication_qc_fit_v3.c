/*
 * Numerically guarded Gaussian fitters for replication_qc_v3.py.
 *
 * This backend deliberately exposes only single-crop primitives.  Python owns
 * deterministic multi-start selection and bootstrap orchestration, which keeps
 * the statistical procedure auditable.  Both models use the same optimizer.
 * Normal equations are column-scaled before solving and pivots are assessed
 * relative to the scaled matrix norm; a near-singular system is therefore an
 * explicit fit failure rather than a source of meaningless covariance output.
 */

#include <math.h>
#include <stddef.h>
#include <string.h>
#include <float.h>

#define NPAR1 8
#define NPAR2 12
#define MAXPAR 12
#define DEFAULT_MAX_ITER 200
#define REL_PIVOT_TOL 1e-12

static inline double sqr(double x) { return x * x; }

static int center_supported(double cz, double cy, double cx,
                            const double *z, const double *y, const double *x,
                            int n)
{
    int iz = (int)lround(cz), iy = (int)lround(cy), ix = (int)lround(cx);
    for (int i = 0; i < n; ++i) {
        if ((int)z[i] == iz && (int)y[i] == iy && (int)x[i] == ix) return 1;
    }
    return 0;
}

static inline double gaussian(double amp, double cz, double cy, double cx,
                              double sz, double sy, double sx,
                              double z, double y, double x)
{
    double dz = (z - cz) / sz;
    double dy = (y - cy) / sy;
    double dx = (x - cx) / sx;
    return amp * exp(-0.5 * (dz * dz + dy * dy + dx * dx));
}

static void clamp(double *p, const double *lo, const double *hi, int n)
{
    for (int i = 0; i < n; ++i) {
        if (isfinite(lo[i]) && p[i] < lo[i]) p[i] = lo[i];
        if (isfinite(hi[i]) && p[i] > hi[i]) p[i] = hi[i];
    }
}

/* Gaussian elimination with a scale-relative pivot test. */
static int solve_relative(const double *a, const double *b, int n, double *x)
{
    double aug[MAXPAR][MAXPAR + 1];
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = a[i * n + j];
            norm = fmax(norm, fabs(aug[i][j]));
        }
        aug[i][n] = b[i];
    }
    if (!(norm > 0.0) || !isfinite(norm)) return 0;

    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        for (int row = col + 1; row < n; ++row) {
            if (fabs(aug[row][col]) > fabs(aug[pivot_row][col])) pivot_row = row;
        }
        if (fabs(aug[pivot_row][col]) <= REL_PIVOT_TOL * norm) return 0;
        if (pivot_row != col) {
            for (int j = col; j <= n; ++j) {
                double tmp = aug[col][j];
                aug[col][j] = aug[pivot_row][j];
                aug[pivot_row][j] = tmp;
            }
        }
        for (int row = col + 1; row < n; ++row) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; ++j) aug[row][j] -= factor * aug[col][j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        if (fabs(aug[i][i]) <= REL_PIVOT_TOL * norm) return 0;
        double value = aug[i][n];
        for (int j = i + 1; j < n; ++j) value -= aug[i][j] * x[j];
        x[i] = value / aug[i][i];
        if (!isfinite(x[i])) return 0;
    }
    return 1;
}

/* Solve damped normal equations after unit-diagonal column scaling. */
static int solve_scaled_normal(const double *jtj, const double *jtr,
                               int n, double lambda, double *dp)
{
    double scale[MAXPAR], normalized[MAXPAR * MAXPAR], rhs[MAXPAR], q[MAXPAR];
    double max_diag = 0.0;
    for (int i = 0; i < n; ++i) max_diag = fmax(max_diag, fabs(jtj[i * n + i]));
    if (!(max_diag > 0.0) || !isfinite(max_diag)) return 0;
    for (int i = 0; i < n; ++i) {
        double diagonal = jtj[i * n + i];
        if (!(diagonal > DBL_MIN) || !isfinite(diagonal)) return 0;
        scale[i] = sqrt(diagonal);
        rhs[i] = jtr[i] / scale[i];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            normalized[i * n + j] = jtj[i * n + j] / (scale[i] * scale[j]);
        }
        normalized[i * n + i] += lambda;
    }
    if (!solve_relative(normalized, rhs, n, q)) return 0;
    for (int i = 0; i < n; ++i) dp[i] = q[i] / scale[i];
    return 1;
}

static int normalized_information_ok(const double *jtj, int n)
{
    double scale[MAXPAR], normalized[MAXPAR * MAXPAR], zeros[MAXPAR] = {0};
    double solution[MAXPAR] = {0}, max_diag = 0.0;
    for (int i = 0; i < n; ++i) max_diag = fmax(max_diag, fabs(jtj[i * n + i]));
    if (!(max_diag > 0.0) || !isfinite(max_diag)) return 0;
    for (int i = 0; i < n; ++i) {
        double diagonal = jtj[i * n + i];
        if (!(diagonal > DBL_MIN) || !isfinite(diagonal)) return 0;
        scale[i] = sqrt(diagonal);
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            normalized[i * n + j] = jtj[i * n + j] / (scale[i] * scale[j]);
    return solve_relative(normalized, zeros, n, solution);
}

static int final_information1(const double *p, const double *z,
                              const double *y, const double *x, int n)
{
    double jtj[NPAR1 * NPAR1] = {0};
    for (int i = 0; i < n; ++i) {
        double dz = z[i] - p[1], dy = y[i] - p[2], dx = x[i] - p[3];
        double sz2 = p[4] * p[4], sy2 = p[5] * p[5], sx2 = p[6] * p[6];
        double e = exp(-0.5 * (dz * dz / sz2 + dy * dy / sy2 + dx * dx / sx2));
        double g = p[0] * e;
        double jac[NPAR1] = {
            e, g * dz / sz2, g * dy / sy2, g * dx / sx2,
            g * dz * dz / (p[4] * sz2), g * dy * dy / (p[5] * sy2),
            g * dx * dx / (p[6] * sx2), 1.0
        };
        for (int a = 0; a < NPAR1; ++a)
            for (int b = 0; b < NPAR1; ++b) jtj[a * NPAR1 + b] += jac[a] * jac[b];
    }
    return normalized_information_ok(jtj, NPAR1);
}

static int final_information2(const double *p, const double *z,
                              const double *y, const double *x, int n)
{
    double jtj[NPAR2 * NPAR2] = {0};
    for (int i = 0; i < n; ++i) {
        double dz1 = z[i] - p[2], dy1 = y[i] - p[3], dx1 = x[i] - p[4];
        double dz2 = z[i] - p[5], dy2 = y[i] - p[6], dx2 = x[i] - p[7];
        double sz2 = p[8] * p[8], sy2 = p[9] * p[9], sx2 = p[10] * p[10];
        double e1 = exp(-0.5 * (dz1 * dz1 / sz2 + dy1 * dy1 / sy2 + dx1 * dx1 / sx2));
        double e2 = exp(-0.5 * (dz2 * dz2 / sz2 + dy2 * dy2 / sy2 + dx2 * dx2 / sx2));
        double g1 = p[0] * e1, g2 = p[1] * e2;
        double jac[NPAR2] = {
            e1, e2, g1 * dz1 / sz2, g1 * dy1 / sy2, g1 * dx1 / sx2,
            g2 * dz2 / sz2, g2 * dy2 / sy2, g2 * dx2 / sx2,
            (g1 * dz1 * dz1 + g2 * dz2 * dz2) / (p[8] * sz2),
            (g1 * dy1 * dy1 + g2 * dy2 * dy2) / (p[9] * sy2),
            (g1 * dx1 * dx1 + g2 * dx2 * dx2) / (p[10] * sx2), 1.0
        };
        for (int a = 0; a < NPAR2; ++a)
            for (int b = 0; b < NPAR2; ++b) jtj[a * NPAR2 + b] += jac[a] * jac[b];
    }
    return normalized_information_ok(jtj, NPAR2);
}

static double sse1(const double *p, const double *obs,
                   const double *z, const double *y, const double *x, int n)
{
    if (!center_supported(p[1], p[2], p[3], z, y, x, n)) return INFINITY;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
        double fitted = gaussian(p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                 z[i], y[i], x[i]) + p[7];
        value += sqr(obs[i] - fitted);
    }
    return value;
}

static double sse2(const double *p, const double *obs,
                   const double *z, const double *y, const double *x, int n)
{
    if (!center_supported(p[2], p[3], p[4], z, y, x, n) ||
        !center_supported(p[5], p[6], p[7], z, y, x, n)) return INFINITY;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
        double fitted = gaussian(p[0], p[2], p[3], p[4], p[8], p[9], p[10],
                                 z[i], y[i], x[i]);
        fitted += gaussian(p[1], p[5], p[6], p[7], p[8], p[9], p[10],
                           z[i], y[i], x[i]) + p[11];
        value += sqr(obs[i] - fitted);
    }
    return value;
}

/* Return: 0 converged, 1 did not converge, 2 non-finite, 3 ill-conditioned,
 * -1 invalid input.  Returned parameters/SSE are valid only for rc 0 or 1. */
int fit1_v3(const double *obs, const double *z, const double *y, const double *x,
            int n, const double *p0, const double *lo, const double *hi,
            int max_iter, double *out, double *out_sse, int *out_nfev,
            int *out_condition_failures, int *out_final_condition_ok)
{
    if (n < NPAR1 + 1) return -1;
    if (max_iter <= 0) max_iter = DEFAULT_MAX_ITER;
    double p[NPAR1]; memcpy(p, p0, sizeof(p)); clamp(p, lo, hi, NPAR1);
    double chi2 = sse1(p, obs, z, y, x, n);
    if (!isfinite(chi2)) return -1;
    int nfev = 1, converged = 0, condition_failures = 0;
    double lambda = 1e-3;

    for (int iter = 0; iter < max_iter; ++iter) {
        double jtj[NPAR1 * NPAR1] = {0}, jtr[NPAR1] = {0};
        for (int i = 0; i < n; ++i) {
            double dz = z[i] - p[1], dy = y[i] - p[2], dx = x[i] - p[3];
            double sz2 = p[4] * p[4], sy2 = p[5] * p[5], sx2 = p[6] * p[6];
            double e = exp(-0.5 * (dz * dz / sz2 + dy * dy / sy2 + dx * dx / sx2));
            double g = p[0] * e, residual = obs[i] - (g + p[7]);
            double jac[NPAR1] = {
                e, g * dz / sz2, g * dy / sy2, g * dx / sx2,
                g * dz * dz / (p[4] * sz2),
                g * dy * dy / (p[5] * sy2),
                g * dx * dx / (p[6] * sx2), 1.0
            };
            for (int a = 0; a < NPAR1; ++a) {
                jtr[a] += jac[a] * residual;
                for (int b = 0; b < NPAR1; ++b) jtj[a * NPAR1 + b] += jac[a] * jac[b];
            }
        }
        double scaled_grad = 0.0;
        for (int i = 0; i < NPAR1; ++i) {
            double d = sqrt(fmax(jtj[i * NPAR1 + i], 0.0));
            if (d > 0.0) scaled_grad = fmax(scaled_grad, fabs(jtr[i]) / d);
        }
        if (scaled_grad <= 1e-8 * (1.0 + sqrt(fmax(chi2, 0.0)))) { converged = 1; break; }

        double dp[NPAR1];
        if (!solve_scaled_normal(jtj, jtr, NPAR1, lambda, dp)) {
            ++condition_failures; lambda *= 10.0;
            if (lambda > 1e10) break;
            continue;
        }
        double trial[NPAR1], step = 0.0;
        for (int i = 0; i < NPAR1; ++i) {
            trial[i] = p[i] + dp[i];
        }
        clamp(trial, lo, hi, NPAR1);
        for (int i = 0; i < NPAR1; ++i)
            step = fmax(step, fabs(trial[i] - p[i]) / (1.0 + fabs(p[i])));
        double next = sse1(trial, obs, z, y, x, n); ++nfev;
        if (isfinite(next) && next < chi2) {
            double rel = (chi2 - next) / (fabs(chi2) + 1e-30);
            memcpy(p, trial, sizeof(p)); chi2 = next; lambda = fmax(1e-12, lambda * 0.1);
            if (rel < 1e-8 && step < 1e-6) { converged = 1; break; }
        } else {
            lambda *= 10.0;
            if (lambda > 1e10) break;
        }
    }
    *out_nfev = nfev; *out_condition_failures = condition_failures;
    for (int i = 0; i < NPAR1; ++i) if (!isfinite(p[i])) return 2;
    memcpy(out, p, sizeof(p)); *out_sse = chi2;
    *out_final_condition_ok = final_information1(p, z, y, x, n);
    if (!converged && condition_failures > 0 && lambda > 1e10) return 3;
    return converged ? 0 : 1;
}

int fit2_v3(const double *obs, const double *z, const double *y, const double *x,
            int n, const double *p0, const double *lo, const double *hi,
            int max_iter, double *out, double *out_sse, int *out_nfev,
            int *out_condition_failures, int *out_pinned,
            int *out_final_condition_ok, int *out_xy_center_pinned,
            int *out_lateral_width_pinned, int *out_axial_pinned)
{
    if (n < NPAR2 + 1) return -1;
    if (max_iter <= 0) max_iter = DEFAULT_MAX_ITER;
    double p[NPAR2]; memcpy(p, p0, sizeof(p)); clamp(p, lo, hi, NPAR2);
    double chi2 = sse2(p, obs, z, y, x, n);
    if (!isfinite(chi2)) return -1;
    int nfev = 1, converged = 0, condition_failures = 0;
    double lambda = 1e-3;

    for (int iter = 0; iter < max_iter; ++iter) {
        double jtj[NPAR2 * NPAR2] = {0}, jtr[NPAR2] = {0};
        for (int i = 0; i < n; ++i) {
            double dz1 = z[i] - p[2], dy1 = y[i] - p[3], dx1 = x[i] - p[4];
            double dz2 = z[i] - p[5], dy2 = y[i] - p[6], dx2 = x[i] - p[7];
            double sz2 = p[8] * p[8], sy2 = p[9] * p[9], sx2 = p[10] * p[10];
            double e1 = exp(-0.5 * (dz1 * dz1 / sz2 + dy1 * dy1 / sy2 + dx1 * dx1 / sx2));
            double e2 = exp(-0.5 * (dz2 * dz2 / sz2 + dy2 * dy2 / sy2 + dx2 * dx2 / sx2));
            double g1 = p[0] * e1, g2 = p[1] * e2;
            double residual = obs[i] - (g1 + g2 + p[11]);
            double jac[NPAR2] = {
                e1, e2,
                g1 * dz1 / sz2, g1 * dy1 / sy2, g1 * dx1 / sx2,
                g2 * dz2 / sz2, g2 * dy2 / sy2, g2 * dx2 / sx2,
                (g1 * dz1 * dz1 + g2 * dz2 * dz2) / (p[8] * sz2),
                (g1 * dy1 * dy1 + g2 * dy2 * dy2) / (p[9] * sy2),
                (g1 * dx1 * dx1 + g2 * dx2 * dx2) / (p[10] * sx2), 1.0
            };
            for (int a = 0; a < NPAR2; ++a) {
                jtr[a] += jac[a] * residual;
                for (int b = 0; b < NPAR2; ++b) jtj[a * NPAR2 + b] += jac[a] * jac[b];
            }
        }
        double scaled_grad = 0.0;
        for (int i = 0; i < NPAR2; ++i) {
            double d = sqrt(fmax(jtj[i * NPAR2 + i], 0.0));
            if (d > 0.0) scaled_grad = fmax(scaled_grad, fabs(jtr[i]) / d);
        }
        if (scaled_grad <= 1e-8 * (1.0 + sqrt(fmax(chi2, 0.0)))) { converged = 1; break; }

        double dp[NPAR2];
        if (!solve_scaled_normal(jtj, jtr, NPAR2, lambda, dp)) {
            ++condition_failures; lambda *= 10.0;
            if (lambda > 1e10) break;
            continue;
        }
        double trial[NPAR2], step = 0.0;
        for (int i = 0; i < NPAR2; ++i) trial[i] = p[i] + dp[i];
        clamp(trial, lo, hi, NPAR2);
        for (int i = 0; i < NPAR2; ++i)
            step = fmax(step, fabs(trial[i] - p[i]) / (1.0 + fabs(p[i])));
        double next = sse2(trial, obs, z, y, x, n); ++nfev;
        if (isfinite(next) && next < chi2) {
            double rel = (chi2 - next) / (fabs(chi2) + 1e-30);
            memcpy(p, trial, sizeof(p)); chi2 = next; lambda = fmax(1e-12, lambda * 0.1);
            if (rel < 1e-8 && step < 1e-6) { converged = 1; break; }
        } else {
            lambda *= 10.0;
            if (lambda > 1e10) break;
        }
    }
    *out_nfev = nfev; *out_condition_failures = condition_failures;
    for (int i = 0; i < NPAR2; ++i) if (!isfinite(p[i])) return 2;
    memcpy(out, p, sizeof(p)); *out_sse = chi2;
    int pinned = 0;
    for (int i = 0; i < NPAR2; ++i) {
        double span = (isfinite(lo[i]) && isfinite(hi[i])) ? hi[i] - lo[i] : 0.0;
        double eps = fmax(1e-8, 0.01 * span);
        if ((isfinite(lo[i]) && p[i] - lo[i] <= eps) ||
            (isfinite(hi[i]) && hi[i] - p[i] <= eps)) pinned = 1;
    }
    *out_pinned = pinned;
    *out_final_condition_ok = final_information2(p, z, y, x, n);
    int xy_center_pinned = 0, lateral_width_pinned = 0, axial_pinned = 0;
    int xy_center_indices[4] = {3, 4, 6, 7};
    int lateral_width_indices[2] = {9, 10};
    int axial_indices[3] = {2, 5, 8};
    for (int k = 0; k < 4; ++k) {
        int i = xy_center_indices[k];
        double eps = fmax(1e-8, 0.01 * (hi[i] - lo[i]));
        if (p[i] - lo[i] <= eps || hi[i] - p[i] <= eps) xy_center_pinned = 1;
    }
    for (int k = 0; k < 2; ++k) {
        int i = lateral_width_indices[k];
        double eps = fmax(1e-8, 0.01 * (hi[i] - lo[i]));
        if (p[i] - lo[i] <= eps || hi[i] - p[i] <= eps) lateral_width_pinned = 1;
    }
    for (int k = 0; k < 3; ++k) {
        int i = axial_indices[k];
        double eps = fmax(1e-8, 0.01 * (hi[i] - lo[i]));
        if (p[i] - lo[i] <= eps || hi[i] - p[i] <= eps) axial_pinned = 1;
    }
    *out_xy_center_pinned = xy_center_pinned;
    *out_lateral_width_pinned = lateral_width_pinned;
    *out_axial_pinned = axial_pinned;
    if (!converged && condition_failures > 0 && lambda > 1e10) return 3;
    return converged ? 0 : 1;
}
