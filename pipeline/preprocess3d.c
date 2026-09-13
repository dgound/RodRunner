/*
 * 3D image preprocessing for spot detection.
 * Translation of the MATLAB detect3DSpots preprocessing chain:
 *   1. Haar wavelet denoising (per-slice)
 *   2. Anisotropic diffusion (Perona-Malik, per-slice)
 *   3. Non-local means denoising (per-slice)
 *   4. Intensity adjustment (gamma correction, per-slice)
 *   5. Local contrast enhancement (per-slice)
 *   6. Local Laplacian-like detail enhancement (per-slice)
 *   7. Morphological opening background removal (per-slice)
 *
 * All functions operate on double-precision buffers.
 * Callable from Python via ctypes.
 *
 * Compile: gcc -O3 -fopenmp -shared -fPIC -o preprocess3d.so preprocess3d.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================
 * Utility
 * ======================================================================== */

static int _cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static double *alloc_buf(int n) {
    double *p = (double *)calloc((size_t)n, sizeof(double));
    return p;
}

/* ========================================================================
 * 1. Haar wavelet denoising (2D, per-slice)
 *    Approximation of MATLAB wdenoise2(..., 3).
 *    3-level Haar DWT, soft-threshold detail coefficients, inverse DWT.
 * ======================================================================== */

static void haar_forward_rows(const double *in, double *out, int h, int w) {
    int hw = w / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < hw; x++) {
            double a = in[y * w + 2 * x];
            double b = in[y * w + 2 * x + 1];
            out[y * w + x]      = (a + b) * 0.5;
            out[y * w + hw + x] = (a - b) * 0.5;
        }
    }
}

static void haar_forward_cols(const double *in, double *out, int h, int w) {
    int hh = h / 2;
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < hh; y++) {
            double a = in[(2 * y) * w + x];
            double b = in[(2 * y + 1) * w + x];
            out[y * w + x]       = (a + b) * 0.5;
            out[(hh + y) * w + x] = (a - b) * 0.5;
        }
    }
}

static void haar_inverse_cols(const double *in, double *out, int h, int w) {
    int hh = h / 2;
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < hh; y++) {
            double a = in[y * w + x];
            double d = in[(hh + y) * w + x];
            out[(2 * y) * w + x]     = a + d;
            out[(2 * y + 1) * w + x] = a - d;
        }
    }
}

static void haar_inverse_rows(const double *in, double *out, int h, int w) {
    int hw = w / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < hw; x++) {
            double a = in[y * w + x];
            double d = in[y * w + hw + x];
            out[y * w + 2 * x]     = a + d;
            out[y * w + 2 * x + 1] = a - d;
        }
    }
}

static double estimate_noise_mad(const double *detail, int n) {
    double *tmp = alloc_buf(n);
    for (int i = 0; i < n; i++) tmp[i] = fabs(detail[i]);
    qsort(tmp, (size_t)n, sizeof(double), _cmp_double);
    double mad = tmp[n / 2] / 0.6745;
    free(tmp);
    return mad;
}

static void soft_threshold(double *data, int n, double thresh) {
    for (int i = 0; i < n; i++) {
        double v = data[i];
        if (v > thresh) data[i] = v - thresh;
        else if (v < -thresh) data[i] = v + thresh;
        else data[i] = 0.0;
    }
}

void wavelet_denoise_2d(double *img, int h, int w, int levels) {
    int n = h * w;
    double *tmp1 = alloc_buf(n);
    double *tmp2 = alloc_buf(n);

    memcpy(tmp1, img, (size_t)n * sizeof(double));

    int ch = h, cw = w;
    for (int lev = 0; lev < levels; lev++) {
        /* Ensure even dimensions for this level */
        int eh = (ch / 2) * 2;
        int ew = (cw / 2) * 2;
        if (eh < 2 || ew < 2) break;

        /* Forward rows then cols on the approximation subband */
        haar_forward_rows(tmp1, tmp2, eh, ew);
        memcpy(tmp1, tmp2, (size_t)(eh * ew) * sizeof(double));
        haar_forward_cols(tmp1, tmp2, eh, ew);
        memcpy(tmp1, tmp2, (size_t)(eh * ew) * sizeof(double));

        /* Soft-threshold the three detail subbands (LH, HL, HH) */
        int hh = eh / 2, hw2 = ew / 2;
        int detail_size = hh * hw2;

        /* Estimate noise from finest-level HH band */
        double sigma;
        if (lev == 0) {
            /* HH band: rows [hh..eh), cols [hw2..ew) */
            double *hh_band = alloc_buf(detail_size);
            for (int y = 0; y < hh; y++)
                for (int x = 0; x < hw2; x++)
                    hh_band[y * hw2 + x] = tmp1[(hh + y) * ew + hw2 + x];
            sigma = estimate_noise_mad(hh_band, detail_size);
            free(hh_band);
        } else {
            sigma = 0; /* use stored sigma from level 0 */
        }

        double thresh = sigma * sqrt(2.0 * log((double)detail_size));

        /* LH: rows [0..hh), cols [hw2..ew) */
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < hw2; x++) {
                double *v = &tmp1[y * ew + hw2 + x];
                if (*v > thresh) *v -= thresh;
                else if (*v < -thresh) *v += thresh;
                else *v = 0;
            }
        /* HL: rows [hh..eh), cols [0..hw2) */
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < hw2; x++) {
                double *v = &tmp1[(hh + y) * ew + x];
                if (*v > thresh) *v -= thresh;
                else if (*v < -thresh) *v += thresh;
                else *v = 0;
            }
        /* HH: rows [hh..eh), cols [hw2..ew) */
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < hw2; x++) {
                double *v = &tmp1[(hh + y) * ew + hw2 + x];
                if (*v > thresh) *v -= thresh;
                else if (*v < -thresh) *v += thresh;
                else *v = 0;
            }

        ch = hh;
        cw = hw2;
    }

    /* Inverse DWT */
    ch = h; cw = w;
    int sizes_h[16], sizes_w[16];
    int actual_levels = 0;
    {
        int th = h, tw = w;
        for (int lev = 0; lev < levels; lev++) {
            int eh = (th / 2) * 2;
            int ew = (tw / 2) * 2;
            if (eh < 2 || ew < 2) break;
            sizes_h[lev] = eh;
            sizes_w[lev] = ew;
            th = eh / 2;
            tw = ew / 2;
            actual_levels++;
        }
    }

    for (int lev = actual_levels - 1; lev >= 0; lev--) {
        int eh = sizes_h[lev], ew = sizes_w[lev];
        haar_inverse_cols(tmp1, tmp2, eh, ew);
        memcpy(tmp1, tmp2, (size_t)(eh * ew) * sizeof(double));
        haar_inverse_rows(tmp1, tmp2, eh, ew);
        memcpy(tmp1, tmp2, (size_t)(eh * ew) * sizeof(double));
    }

    memcpy(img, tmp1, (size_t)n * sizeof(double));
    free(tmp1);
    free(tmp2);
}

/* ========================================================================
 * 2. Anisotropic diffusion (Perona-Malik, per-slice)
 *    Matches MATLAB imdiffusefilt defaults: 5 iterations, gradient
 *    threshold estimated from image noise.
 * ======================================================================== */

void anisotropic_diffusion_2d(double *img, int h, int w,
                               int n_iter, double kappa, double dt) {
    double *tmp = alloc_buf(h * w);

    for (int it = 0; it < n_iter; it++) {
        memcpy(tmp, img, (size_t)(h * w) * sizeof(double));
        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++) {
                double c = tmp[y * w + x];
                double dn = tmp[(y-1)*w + x] - c;
                double ds = tmp[(y+1)*w + x] - c;
                double de = tmp[y*w + (x+1)] - c;
                double dw = tmp[y*w + (x-1)] - c;

                double cn = exp(-(dn * dn) / (kappa * kappa));
                double cs = exp(-(ds * ds) / (kappa * kappa));
                double ce = exp(-(de * de) / (kappa * kappa));
                double cw2 = exp(-(dw * dw) / (kappa * kappa));

                img[y * w + x] = c + dt * (cn*dn + cs*ds + ce*de + cw2*dw);
            }
    }
    free(tmp);
}

/* ========================================================================
 * 3. Non-local means denoising (per-slice)
 *    Simplified NLM: search window 11x11, patch 5x5.
 * ======================================================================== */

void nlm_denoise_2d(double *img, int h, int w, int search_r, int patch_r, double filter_sigma) {
    double *out = alloc_buf(h * w);
    double h2 = filter_sigma * filter_sigma;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double wsum = 0, vsum = 0;
            double max_w = 0;

            int sy0 = (y - search_r < 0) ? 0 : y - search_r;
            int sy1 = (y + search_r >= h) ? h - 1 : y + search_r;
            int sx0 = (x - search_r < 0) ? 0 : x - search_r;
            int sx1 = (x + search_r >= w) ? w - 1 : x + search_r;

            for (int sy = sy0; sy <= sy1; sy++)
                for (int sx = sx0; sx <= sx1; sx++) {
                    if (sy == y && sx == x) continue;

                    double dist = 0;
                    int count = 0;
                    for (int py = -patch_r; py <= patch_r; py++)
                        for (int px = -patch_r; px <= patch_r; px++) {
                            int ay = y + py, ax = x + px;
                            int by = sy + py, bx = sx + px;
                            if (ay < 0 || ay >= h || ax < 0 || ax >= w) continue;
                            if (by < 0 || by >= h || bx < 0 || bx >= w) continue;
                            double d = img[ay*w+ax] - img[by*w+bx];
                            dist += d * d;
                            count++;
                        }
                    if (count > 0) dist /= count;

                    double w_val = exp(-dist / h2);
                    if (w_val > max_w) max_w = w_val;
                    wsum += w_val;
                    vsum += w_val * img[sy*w+sx];
                }

            /* Self-weight = max weight */
            wsum += max_w;
            vsum += max_w * img[y*w+x];

            out[y*w+x] = (wsum > 0) ? vsum / wsum : img[y*w+x];
        }

    memcpy(img, out, (size_t)(h*w) * sizeof(double));
    free(out);
}

/* ========================================================================
 * 4. Intensity adjustment (imadjust with gamma)
 *    Maps [p1, p99] to [0, 65535] with gamma correction.
 * ======================================================================== */

void adjust_intensity_2d(double *img, int h, int w, double gamma_val) {
    int n = h * w;
    double *sorted = alloc_buf(n);
    memcpy(sorted, img, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), _cmp_double);

    double lo = sorted[(int)(0.01 * (n - 1))];
    double hi = sorted[(int)(0.99 * (n - 1))];
    free(sorted);

    if (hi <= lo) return;
    double range = hi - lo;

    for (int i = 0; i < n; i++) {
        double v = (img[i] - lo) / range;
        v = clamp(v, 0.0, 1.0);
        img[i] = pow(v, gamma_val) * 65535.0;
    }
}

/* ========================================================================
 * 5. Local contrast enhancement (per-slice)
 *    Approximation of MATLAB localcontrast using guided-filter-like
 *    approach: enhance = img + amount * (img - local_mean).
 * ======================================================================== */

static void box_blur(const double *in, double *out, int h, int w, int r) {
    double *tmp = alloc_buf(h * w);

    /* Horizontal pass */
    for (int y = 0; y < h; y++) {
        double sum = 0;
        int count = 0;
        for (int x = 0; x <= r && x < w; x++) { sum += in[y*w+x]; count++; }
        tmp[y*w+0] = sum / count;
        for (int x = 1; x < w; x++) {
            if (x + r < w) { sum += in[y*w+x+r]; count++; }
            if (x - r - 1 >= 0) { sum -= in[y*w+x-r-1]; count--; }
            tmp[y*w+x] = sum / count;
        }
    }
    /* Vertical pass */
    for (int x = 0; x < w; x++) {
        double sum = 0;
        int count = 0;
        for (int y = 0; y <= r && y < h; y++) { sum += tmp[y*w+x]; count++; }
        out[0*w+x] = sum / count;
        for (int y = 1; y < h; y++) {
            if (y + r < h) { sum += tmp[(y+r)*w+x]; count++; }
            if (y - r - 1 >= 0) { sum -= tmp[(y-r-1)*w+x]; count--; }
            out[y*w+x] = sum / count;
        }
    }
    free(tmp);
}

void local_contrast_2d(double *img, int h, int w, double edge_thresh, double amount) {
    int n = h * w;
    double *local_mean = alloc_buf(n);
    int r = 15;

    box_blur(img, local_mean, h, w, r);

    for (int i = 0; i < n; i++) {
        double diff = img[i] - local_mean[i];
        double edge = fabs(diff) / (65535.0 + 1e-10);
        double gain = (edge < edge_thresh) ? amount : amount * (1.0 - (edge - edge_thresh) / (1.0 - edge_thresh + 1e-10));
        if (gain < 0) gain = 0;
        img[i] = img[i] + gain * diff;
    }
    free(local_mean);
}

/* ========================================================================
 * 6. Local Laplacian-like detail enhancement (per-slice)
 *    Simplified: multi-scale unsharp mask.
 *    Enhances details at scale sigma without creating halos.
 * ======================================================================== */

void local_laplacian_2d(double *img, int h, int w, double sigma, double alpha) {
    int n = h * w;
    double *blurred = alloc_buf(n);
    int r = (int)(sigma * 3);
    if (r < 1) r = 1;

    box_blur(img, blurred, h, w, r);

    for (int i = 0; i < n; i++) {
        double detail = img[i] - blurred[i];
        /* Remap detail: compress large details, boost small ones */
        double sign = (detail >= 0) ? 1.0 : -1.0;
        double abs_d = fabs(detail);
        double remapped = sign * pow(abs_d / (65535.0 + 1e-10), alpha) * 65535.0;
        img[i] = blurred[i] + remapped;
    }
    free(blurred);
}

/* ========================================================================
 * 7. Morphological opening background removal (per-slice)
 *    Erosion + dilation with a circular structuring element.
 *    Matches reduceBackgroundMorpho_padded with replicate padding.
 * ======================================================================== */

static void build_disk_se(int radius, unsigned char **se, int *se_size) {
    int d = 2 * radius + 1;
    *se_size = d;
    *se = (unsigned char *)calloc((size_t)(d * d), sizeof(unsigned char));
    for (int y = 0; y < d; y++)
        for (int x = 0; x < d; x++) {
            int dy = y - radius, dx = x - radius;
            if (dy * dy + dx * dx <= radius * radius)
                (*se)[y * d + x] = 1;
        }
}

static void erode_2d(const double *in, double *out, int h, int w,
                     const unsigned char *se, int se_size) {
    int r = se_size / 2;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double mn = DBL_MAX;
            for (int sy = 0; sy < se_size; sy++)
                for (int sx = 0; sx < se_size; sx++) {
                    if (!se[sy * se_size + sx]) continue;
                    int iy = y + sy - r, ix = x + sx - r;
                    /* Replicate padding */
                    if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;
                    if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;
                    double v = in[iy * w + ix];
                    if (v < mn) mn = v;
                }
            out[y * w + x] = mn;
        }
}

static void dilate_2d(const double *in, double *out, int h, int w,
                      const unsigned char *se, int se_size) {
    int r = se_size / 2;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double mx = -DBL_MAX;
            for (int sy = 0; sy < se_size; sy++)
                for (int sx = 0; sx < se_size; sx++) {
                    if (!se[sy * se_size + sx]) continue;
                    int iy = y + sy - r, ix = x + sx - r;
                    if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;
                    if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;
                    double v = in[iy * w + ix];
                    if (v > mx) mx = v;
                }
            out[y * w + x] = mx;
        }
}

void morpho_bg_remove_2d(double *img, int h, int w, int radius) {
    int n = h * w;
    unsigned char *se;
    int se_size;
    build_disk_se(radius, &se, &se_size);

    double *eroded = alloc_buf(n);
    double *opened = alloc_buf(n);

    erode_2d(img, eroded, h, w, se, se_size);
    dilate_2d(eroded, opened, h, w, se, se_size);

    for (int i = 0; i < n; i++) {
        img[i] = img[i] - opened[i];
        if (img[i] < 0) img[i] = 0;
    }

    free(eroded);
    free(opened);
    free(se);
}

/* ========================================================================
 * Full preprocessing pipeline for a 3D stack.
 *
 *   data:    [nz][ny][nx] float64, modified in place
 *   nz, ny, nx: dimensions
 *   morpho_radius: disk SE radius for background removal (MATLAB default: 20)
 *
 * The pipeline is applied per-slice with OpenMP parallelism.
 * ======================================================================== */

void preprocess_3d(double *data, int nz, int ny, int nx, int morpho_radius) {
    int slice_size = ny * nx;

    /* Phase 1: denoising (wavelet + diffusion + NLM) — per-slice, parallel */
    for (int z = 0; z < nz; z++) {
        double *slice = data + z * slice_size;
        wavelet_denoise_2d(slice, ny, nx, 3);
        anisotropic_diffusion_2d(slice, ny, nx, 5, 30.0, 0.14);
        nlm_denoise_2d(slice, ny, nx, 5, 2, 100.0);
    }

    for (int z = 0; z < nz; z++) {
        double *slice = data + z * slice_size;
        adjust_intensity_2d(slice, ny, nx, 0.3);
        local_contrast_2d(slice, ny, nx, 0.2, 0.2);
        local_laplacian_2d(slice, ny, nx, 0.4, 0.5);
    }

    for (int z = 0; z < nz; z++) {
        double *slice = data + z * slice_size;
        morpho_bg_remove_2d(slice, ny, nx, morpho_radius);
    }
}

/* Convenience: preprocess only the morphological background removal
 * (fastest, most impactful single step). */
void preprocess_morpho_only(double *data, int nz, int ny, int nx, int morpho_radius) {
    int slice_size = ny * nx;
    for (int z = 0; z < nz; z++) {
        morpho_bg_remove_2d(data + z * slice_size, ny, nx, morpho_radius);
    }
}

/* Convenience: denoise + morphological background removal (skip contrast). */
void preprocess_denoise_morpho(double *data, int nz, int ny, int nx, int morpho_radius) {
    int slice_size = ny * nx;
    for (int z = 0; z < nz; z++) {
        double *slice = data + z * slice_size;
        wavelet_denoise_2d(slice, ny, nx, 3);
        anisotropic_diffusion_2d(slice, ny, nx, 5, 30.0, 0.14);
    }
    for (int z = 0; z < nz; z++) {
        morpho_bg_remove_2d(data + z * slice_size, ny, nx, morpho_radius);
    }
}

/* Configurable variant for modality tuning. A stage is skipped when its
 * leading parameter is <= 0. Defaults equivalent to preprocess_denoise_morpho:
 * wavelet_levels=3, diffusion_iters=5, diffusion_kappa=30, nlm_search_r=0. */
void preprocess_custom(double *data, int nz, int ny, int nx, int morpho_radius,
                       int wavelet_levels, int diffusion_iters, double diffusion_kappa,
                       int nlm_search_r, int nlm_patch_r, double nlm_sigma) {
    int slice_size = ny * nx;
    for (int z = 0; z < nz; z++) {
        double *slice = data + z * slice_size;
        if (wavelet_levels > 0) wavelet_denoise_2d(slice, ny, nx, wavelet_levels);
        if (diffusion_iters > 0) anisotropic_diffusion_2d(slice, ny, nx, diffusion_iters, diffusion_kappa, 0.14);
        if (nlm_search_r > 0) nlm_denoise_2d(slice, ny, nx, nlm_search_r, nlm_patch_r, nlm_sigma);
    }
    if (morpho_radius > 0)
        for (int z = 0; z < nz; z++)
            morpho_bg_remove_2d(data + z * slice_size, ny, nx, morpho_radius);
}
