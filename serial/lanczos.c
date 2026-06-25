#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lanczos.h"
#include "linalg.h"

/*
 * lanczos.c
 *
 * Lanczos eigenvalue estimation for AM̃^{-1} where:
 *   A   = 2D Poisson matrix (5-point stencil, implicit)
 *   M̃   = Block Jacobi preconditioner (Thomas factored diagonal blocks)
 */

/* ------------------------------------------------------------------ */
/* Local Block Jacobi apply (uses Thomas factors from pci_smooth)      */
/* ------------------------------------------------------------------ */

//This is a local function to apply the block Jacobi preconditioner using the Thomas factors computed in the pci_smooth function. 
static void bj_apply_local(int n_int, double h, const double *td, const double *tl, const double *r, double *z)
{
    double h2inv = 1.0 / (h * h);
    double sub  = -h2inv;

    //This loop applies the block Jacobi preconditioner to the input vector r, storing the result in z. It loops over each block (grid row), 
    //performs forward substitution to solve L w = r for the current block, and then back substitution to solve U z = w. The Thomas factors td and 
    //tl are used for the forward and back substitution steps.
    for (int blk = 0; blk < n_int; blk++) {
        const double *rb = r + blk * n_int;
        double *zb = z + blk * n_int;

        // Forward substitution 
        zb[0] = rb[0];
        for (int i = 1; i < n_int; i++)
            zb[i] = rb[i] - tl[i] * zb[i-1];

        // Back substitution 
        zb[n_int-1] /= td[n_int-1];
        for (int i = n_int-2; i >= 0; i--)
            zb[i] = (zb[i] - sub * zb[i+1]) / td[i];
    }
}

/* ------------------------------------------------------------------ */
/* Apply B = A * M̃^{-1}                                               */
/* ------------------------------------------------------------------ */

//This just applies the operator B = A * M̃^{-1} to a vector v.
static void apply_B(int n_int, double h, const double *td, const double *tl, const double *v, double *out)
{
    int N = n_int * n_int;
    double *w = malloc(N * sizeof(double));

    bj_apply_local(n_int, h, td, tl, v, w);
    matvec_poisson2d(n_int, h, w, out);

    free(w);
}

/* ------------------------------------------------------------------ */
/* Symmetric tridiagonal QR eigenvalue solver                          */
/* ------------------------------------------------------------------ */

static void tridiag_qr(int n, double *diag, double *offdiag)
{
    //max no. of iterations.
    int max_iter = 100 * n;

    //loop until convergence or max iterations reached. 
    for (int iter = 0; iter < max_iter; iter++) {

        // Find largest unreduced block, look for small offdiag entry 
        int m = n - 1;
        while (m > 0) {
            double tol = 1e-14 * (fabs(diag[m-1]) + fabs(diag[m]));
            if (fabs(offdiag[m-1]) <= tol) {
                offdiag[m-1] = 0.0;
                break;
            }
            m--;
        }
        if (m == 0) break;   // fully converged 

        // Find start of unreduced block 
        int l = m - 1;
        while (l > 0) {
            double tol = 1e-14 * (fabs(diag[l-1]) + fabs(diag[l]));
            if (fabs(offdiag[l-1]) <= tol) {
                offdiag[l-1] = 0.0;
                break;
            }
            l--;
        }

        // Wilkinson shift: eigenvalue of bottom 2x2 closer to diag[m] 
        double d   = (diag[m-1] - diag[m]) / 2.0;
        double sig = diag[m] - offdiag[m-1]*offdiag[m-1] / (d + copysign(sqrt(d*d + offdiag[m-1]*offdiag[m-1]), d));

        // Implicit QR step on submatrix [l..m] 
        double x = diag[l] - sig;
        double z = offdiag[l];

        for (int k = l; k < m; k++) {
            // Compute Givens rotation to zero z 
            double r = sqrt(x*x + z*z);
            double cs = x / r;
            double sn = z / r;

            // Apply rotation from the left and right 
            if (k > l) offdiag[k-1] = r;

            double q = cs*offdiag[k] + sn*(diag[k+1] - diag[k]);
            diag[k] = diag[k] + sn*(sn*q - cs*offdiag[k]) / (1.0 + cs);
            /* Simplified symmetric update */
            double tmp = diag[k];
            diag[k] = cs*cs*tmp + 2.0*cs*sn*offdiag[k] + sn*sn*diag[k+1];
            diag[k+1] = sn*sn*tmp - 2.0*cs*sn*offdiag[k] + cs*cs*diag[k+1];
            offdiag[k] = cs*sn*(diag[k+1] - tmp) + (cs*cs - sn*sn)*offdiag[k];

            if (k < m - 1) {
                x = offdiag[k];
                z = sn * offdiag[k+1];
                offdiag[k+1] *= cs;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Lanczos iteration                                                   */
/* ------------------------------------------------------------------ */

/*
 * lanczos_eigenvalues
 *
 * Runs k steps of the Lanczos algorithm on B = AM̃^{-1} starting from
 * a random unit vector. Builds the k×k tridiagonal matrix T, then
 * finds its eigenvalues using tridiag_qr, and returns the min and max.
 *
 * Lanczos recurrence (j = 1..k):
 *   z_j     = B * q_j
 *   α_j     = q_j^T * z_j
 *   r_j     = z_j - α_j * q_j - β_j * q_{j-1}
 *   β_{j+1} = ||r_j||
 *   q_{j+1} = r_j / β_{j+1}
 *
 * where β_1 = 0 and q_0 = 0.
 *
 * The tridiagonal T has diagonal α_1..α_k and off-diagonal β_2..β_k.
 */
void lanczos_eigenvalues(int n_int, double h, const double *td, const double *tl, int k, double *lam_min, double *lam_max)
{
    int N = n_int * n_int;

    // Clamp k to problem size 
    if (k > N) k = N;

    // Tridiagonal entries 
    double *alpha = malloc(k * sizeof(double));
    double *beta = calloc(k, sizeof(double));    

    // Lanczos vectors: only need current and previous 
    double *q_prev  = calloc(N, sizeof(double));
    double *q_curr  = calloc(N, sizeof(double));
    double *z  = malloc(N * sizeof(double));

    // Starting vector: constant unit vector 
    for (int i = 0; i < N; i++) q_curr[i] = 1.0;
    double nrm = vec_norm2(N, q_curr);
    for (int i = 0; i < N; i++) q_curr[i] /= nrm;

    for (int j = 0; j < k; j++) {

        // z = B * q_curr 
        apply_B(n_int, h, td, tl, q_curr, z);

        // α_j = q_curr^T * z 
        alpha[j] = vec_dot(N, q_curr, z);

        // r = z - α_j * q_curr - β_j * q_prev 
        for (int i = 0; i < N; i++)
            z[i] -= alpha[j] * q_curr[i] + (j > 0 ? beta[j] * q_prev[i] : 0.0);

        // β_{j+1} = ||r|| 
        double b = vec_norm2(N, z);

        if (j < k - 1) {
            beta[j+1] = b;

            // q_prev = q_curr, q_curr = r / b 
            double *tmp = q_prev;
            q_prev = q_curr;
            q_curr = tmp;

            if (b < 1e-14) {
                // Invariant subspace found, pad remaining with zeros 
                vec_zero(N, q_curr);
                for (int jj = j+1; jj < k; jj++) {
                    alpha[jj] = 0.0;
                    beta[jj]  = 0.0;
                }
                k = j + 1;
                break;
            }

            for (int i = 0; i < N; i++) q_curr[i] = z[i] / b;
        }
    }

    // Find eigenvalues of k×k tridiagonal T via symmetric QR
    double *diag    = malloc(k * sizeof(double));
    double *offdiag = (k > 1) ? malloc((k-1) * sizeof(double)) : NULL;

    for (int j = 0; j < k; j++)     diag[j]    = alpha[j];
    for (int j = 0; j < k-1; j++)   offdiag[j] = beta[j+1];

    tridiag_qr(k, diag, offdiag);

    // Extract min and max — k is guaranteed > 0
    double mn = (k > 0) ? diag[0] : 1e-10;
    double mx = (k > 0) ? diag[0] : 2.0;
    for (int j = 1; j < k; j++) {
        if (diag[j] < mn) mn = diag[j];
        if (diag[j] > mx) mx = diag[j];
    }

    // Safety clamp: eigenvalues must be positive
    *lam_min = (mn > 1e-10) ? mn : 1e-10;
    *lam_max = mx;

    //Free the dynamically allocated memory.
    free(alpha); 
    free(beta);
    free(q_prev); 
    free(q_curr); 
    free(z);
    free(diag); 
    if (offdiag) free(offdiag);
}