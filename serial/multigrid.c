#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "multigrid.h"
#include "linalg.h"
#include "lanczos.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
 * multigrid.c
 *
 * This file implements the four building blocks of the multigrid V-cycle for the
 * 2D Poisson problem on a uniform interior grid.
 */


/* ------------------------------------------------------------------ */
/* Weighted Jacobi Smoother                                           */
/* ------------------------------------------------------------------ */

//This is our function for doing the weighted jacobi smoother. It takes in the current iterate u, the rhs f, and the parameters for 
//the smoother (number of steps nu and damping weight omega), and it updates u in place by performing nu sweeps of weighted Jacobi 
//iteration. The matrix-vector product Au is computed using the matvec_poisson2d function, which applies the 5-point stencil implicitly 
//without forming the matrix.
void wj_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    int N = n_int * n_int; //Number of interior grid points
    double diag  = 4.0 / (h * h); //Diagonal of A.
    double scale = omega / diag; //Jacobi update scaling factor (omega * D^{-1})

    //Allocate memory for Au
    double *Au = malloc(N * sizeof(double));

    //This loop performs nu sweeps of weighted Jacobi iteration. 
    for (int sweep = 0; sweep < nu; sweep++) {

        ///Here we are computing Au = A*u using the matvec function from linalg.h.
        matvec_poisson2d(n_int, h, u, Au);

        
        //This loop updates each point of u by adding the scaled residual (f - Au) to it. This is the 
        //weighted Jacobi update: u = u + omega*D^{-1}*(f - A*u).
        for (int k = 0; k < N; k++)
            u[k] += scale * (f[k] - Au[k]);
    }

    //We free dynamically allocated memory for Au.
    free(Au);
}


/* ------------------------------------------------------------------ */
/* Gauss-Seidel Smoother                                           */
/* ------------------------------------------------------------------ */
void gs_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    //We don't actually need an omega parameter for Gauss-Seidel, but we include it in the function signature to match the SmootherFn type and 
    //allow for easy switching between smoothers.
    (void)omega;

    //This is just h^2.
    double h2 = h * h;

    //This loop does our Gauss-Seidel forward and backward sweeps.
    for (int sweep = 0; sweep < nu; sweep++) {

        // Forward pass: i = 0..n_int-1, j = 0..n_int-1 
        for (int i = 0; i < n_int; i++) {
            for (int j = 0; j < n_int; j++) {
                double nb = 0.0;
                if (i > 0) nb += u[(i-1)*n_int + j];
                if (i < n_int - 1)  nb += u[(i+1)*n_int + j];
                if (j > 0) nb += u[i*n_int + (j-1)];
                if (j < n_int - 1)  nb += u[i*n_int + (j+1)];
                u[i*n_int + j] = (h2 * f[i*n_int + j] + nb) / 4.0;
            }
        }

        // Backward pass: i = n_int-1..0, j = n_int-1..0 
        for (int i = n_int - 1; i >= 0; i--) {
            for (int j = n_int - 1; j >= 0; j--) {
                double nb = 0.0;
                if (i > 0) nb += u[(i-1)*n_int + j];
                if (i < n_int - 1)  nb += u[(i+1)*n_int + j];
                if (j > 0) nb += u[i*n_int + (j-1)];
                if (j < n_int - 1)  nb += u[i*n_int + (j+1)];
                u[i*n_int + j] = (h2 * f[i*n_int + j] + nb) / 4.0;
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/* Smoother: Red-Black Gauss-Seidel                                    */
/* ------------------------------------------------------------------ */

/*
 * Symmetric Red-Black Gauss-Seidel smoother for A*u = f.
 *
 * Grid points are coloured like a checkerboard:
 *   Red   if (i+j) % 2 == 0
 *   Black if (i+j) % 2 == 1
 *
 * Each of the nu sweeps consists of:
 *   Forward:  update all RED   → update all BLACK
 *   Backward: update all BLACK → update all RED
 */
void rb_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    //Similarly to the Gauss-Seidel smoother, we don't actually need the omega parameter for the red-black Gauss-Seidel, but we include it in 
    //the function signature to match the SmootherFn type and allow for easy switching between smoothers.
    (void)omega;

    //h^2
    double h2 = h * h;

    //This loop performs nu sweeps of the red-black Gauss-Seidel iteration. Each sweep consists of a forward pass where we first update all the red
    //points (where (i+j) is even) and then all the black points (where (i+j) is odd), followed by a backward pass where we update the black points
    // first and then the red points. The updates are done using the 5-point stencil formula for the Poisson problem, similar to the Gauss-Seidel 
    //smoother.
    for (int sweep = 0; sweep < nu; sweep++) {

        //Forward pass: RED then BLACK 
        for (int colour = 0; colour <= 1; colour++) {
            for (int i = 0; i < n_int; i++) {
                for (int j = 0; j < n_int; j++) {
                    if ((i + j) % 2 != colour) continue;
                    double nb = 0.0;
                    if (i > 0) nb += u[(i-1)*n_int + j];
                    if (i < n_int - 1) nb += u[(i+1)*n_int + j];
                    if (j > 0) nb += u[i*n_int + (j-1)];
                    if (j < n_int - 1) nb += u[i*n_int + (j+1)];
                    u[i*n_int + j] = (h2 * f[i*n_int + j] + nb) / 4.0;
                }
            }
        }

        //Backward: BLACK then RED
        for (int colour = 1; colour >= 0; colour--) {
            for (int i = n_int - 1; i >= 0; i--) {
                for (int j = n_int - 1; j >= 0; j--) {
                    if ((i + j) % 2 != colour) continue;
                    double nb = 0.0;
                    if (i > 0) nb += u[(i-1)*n_int + j];
                    if (i < n_int - 1) nb += u[(i+1)*n_int + j];
                    if (j > 0) nb += u[i*n_int + (j-1)];
                    if (j < n_int - 1) nb += u[i*n_int + (j+1)];
                    u[i*n_int + j] = (h2 * f[i*n_int + j] + nb) / 4.0;
                }
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/* Chebyshev Smoother                                                */
/* ------------------------------------------------------------------ */

/*
 * Chebyshev polynomial smoother for A*u = f.
 *
 * Uses a non-stationary three-term recurrence to optimally damp the
 * high-frequency error components over the interval [alpha, beta].
 *
 * Requires two extra work vectors (u_prev and d) of length N.
 */
void cheb_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    (void)omega;

    int N = n_int * n_int;
    double h2 = h * h;
    double beta = 8.0 / h2;             //Spectral radius of A          
    double alpha = beta / 3.0;           //Target upper 2/3 of spectrum 
    double theta = (beta + alpha) / 2.0; //Centre of target interval
    double delta = (beta - alpha) / 2.0; //Half-width of target interval
    double diag = 4.0 / h2;             //Diagonal of A 

    //Dynamically allocate memory for vectors
    double *r      = malloc(N * sizeof(double));
    double *u_prev = malloc(N * sizeof(double));

    //Initialise rho_prev to 1.
    double rho_prev = 1.0;

    //This loop performs the chebyshev smoother.
    for (int k = 1; k <= nu; k++) {

        // r = f - A*u 
        matvec_poisson2d(n_int, h, u, r);
        for (int i = 0; i < N; i++)
            r[i] = f[i] - r[i];

        //This is the case where k=1, where we just do a scaled Jacobi update. For k > 1, we use the three-term recurrence to compute the new
        // iterate.
        if (k == 1) {
            // First step: u = u + (1/theta) * D^{-1} * r 
            double s = 1.0 / (theta * diag);
            for (int i = 0; i < N; i++) {
                u_prev[i] = u[i];
                u[i]     += s * r[i];
            }
            rho_prev = 1.0;
        } else {
            // rho_k = 1 / (2*theta/delta - rho_{k-1})
            double rho_k = 1.0 / (2.0 * theta / delta - rho_prev);

            // u_new = rho_k * (2/delta * D^{-1} * r + u - u_prev) + u_prev
            double s = (2.0 / delta) / diag;
            for (int i = 0; i < N; i++) {
                double u_new = rho_k * (s * r[i] + u[i] - u_prev[i]) + u_prev[i];
                u_prev[i]   = u[i];
                u[i]        = u_new;
            }
            rho_prev = rho_k;
        }
    }

    //Free the dynamically allocated memory.
    free(r);
    free(u_prev);
}

/* ------------------------------------------------------------------ */
/* Successive Over-Relaxation (SOR) Smoother                          */
/* ------------------------------------------------------------------ */
//This is our function for the SSOR smoother, which is very similar to the Gauss-Seidel smoother but includes a damping weight omega and performs
// a forward sweep followed by a backward sweep in each iteration.
void sor_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    //h^2
    double h2 = h * h;

    //We do nu loops.
    for (int sweep = 0; sweep < nu; sweep++) {

        // Forward sweep 
        for (int i = 0; i < n_int; i++) {
            for (int j = 0; j < n_int; j++) {

                int idx = i * n_int + j;

                double nb = 0.0;

                if (i > 0)           nb += u[(i-1)*n_int + j];
                if (i < n_int - 1)   nb += u[(i+1)*n_int + j];
                if (j > 0)           nb += u[i*n_int + (j-1)];
                if (j < n_int - 1)   nb += u[i*n_int + (j+1)];

                double gs =
                    (h2 * f[idx] + nb) / 4.0;

                u[idx] =
                    (1.0 - omega) * u[idx]
                    + omega * gs;
            }
        }

        // Backward sweep 
        for (int i = n_int - 1; i >= 0; i--) {
            for (int j = n_int - 1; j >= 0; j--) {

                int idx = i * n_int + j;

                double nb = 0.0;

                if (i > 0) nb += u[(i-1)*n_int + j];
                if (i < n_int - 1) nb += u[(i+1)*n_int + j];
                if (j > 0) nb += u[i*n_int + (j-1)];
                if (j < n_int - 1) nb += u[i*n_int + (j+1)];

                double gs = (h2 * f[idx] + nb) / 4.0;

                u[idx] = (1.0 - omega) * u[idx] + omega * gs;
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/* Smoother: Preconditioned Chebyshev Iteration (P-CI)                */
/* ------------------------------------------------------------------ */


void pci_smooth(int n_int, double h, double omega, int nu, double *u, const double *f)
{
    //Don't need omega, include it in the function signature to match the SmootherFn type and allow for easy switching between smoothers.
    (void)omega;

    //number of interior points 
    int N = n_int * n_int;
    //1/h^2
    double h2inv = 1.0 / (h * h);
    //Off-diagonal entries of the 1D block tridiagonal T
    double sub = -h2inv;

    //Allocating memory  
    double *td = malloc(n_int * sizeof(double));
    double *tl = malloc(n_int * sizeof(double));

    //Thomas factorisation of the 1D block tridiagonal T, which is the block Jacobi preconditioner.
    for (int i = 0; i < n_int; i++) td[i] = 2.0 * h2inv;
    tl[0] = 0.0;
    for (int i = 1; i < n_int; i++) {
        tl[i]  = sub / td[i-1];
        td[i] -= tl[i] * sub;
    }

    // Eigenvalue estimates via Lanczos 
    double lam_min, lam_max;
    lanczos_eigenvalues(n_int, h, td, tl, 30, &lam_min, &lam_max);

    double d = (lam_max + lam_min) / 2.0;
    double c = (lam_max - lam_min) / 2.0;

    //Allocate memory for vectors.
    double *r = malloc(N * sizeof(double));
    double *z = malloc(N * sizeof(double));
    double *p = calloc(N, sizeof(double));

    double alpha_prev = 1.0 / d;

    // P-CI iterations 
    for (int iter = 1; iter <= nu; iter++) {

        // u = u + α_{i-1} * p_{i-1} 
        for (int k = 0; k < N; k++)
            u[k] += alpha_prev * p[k];

        // r = f - A*u 
        matvec_poisson2d(n_int, h, u, r);
        for (int k = 0; k < N; k++)
            r[k] = f[k] - r[k];

        // z = M̃^{-1} * r  (Block Jacobi Thomas solve) 
        for (int blk = 0; blk < n_int; blk++) {
            const double *rb = r + blk * n_int;
            double *zb = z + blk * n_int;
            zb[0] = rb[0];
            for (int i = 1; i < n_int; i++)
                zb[i] = rb[i] - tl[i] * zb[i-1];
            zb[n_int-1] /= td[n_int-1];
            for (int i = n_int-2; i >= 0; i--)
                zb[i] = (zb[i] - sub * zb[i+1]) / td[i];
        }

        // β and α update 
        double beta  = (iter == 1) ? 0.0 : (alpha_prev * c / 2.0) * (alpha_prev * c / 2.0);
        double alpha = (iter == 1) ? alpha_prev : 1.0 / (d - beta / alpha_prev);

        // p = z + β * p 
        for (int k = 0; k < N; k++)
            p[k] = z[k] + beta * p[k];

        alpha_prev = alpha;
    }

    // Final update
    for (int k = 0; k < N; k++)
        u[k] += alpha_prev * p[k];

    free(td); free(tl);
    free(r);  free(z);  free(p);
}


/* ------------------------------------------------------------------ */
/* This is the restriction section with full-weighting                                         */
/* ------------------------------------------------------------------ */

/*
 * This function implements the full-weighting restriction operator,.
 *
 * Full-weighting restriction from fine grid (n_fine interior points)
 * to coarse grid (n_coarse = (n_fine - 1)/2 interior points).
 *
 * The full-weighting stencil at coarse point (I,J) gathers from the
 * corresponding fine point (2I, 2J) and its 8 neighbours:
 *
 * coarse[I,J] = (1/16) * [  fine[2I-1,2J-1] + 2*fine[2I-1,2J] + fine[2I-1,2J+1]
 *                         + 2*fine[2I,  2J-1] + 4*fine[2I,  2J] + 2*fine[2I,  2J+1]
 *                         + fine[2I+1,2J-1] + 2*fine[2I+1,2J] + fine[2I+1,2J+1] ]
 */
void restrict2d(int n_fine, const double *fine, double *coarse)
{
    //This is the number of points on the coarse grid.
    int n_coarse = (n_fine - 1) / 2;

    //This macro defines the fine grid value at (fi, fj), 0 if out of bounds.
    #define F(fi, fj) \
        (((fi) >= 0 && (fi) < n_fine && (fj) >= 0 && (fj) < n_fine) \
         ? fine[(fi)*n_fine + (fj)] : 0.0)

    //Here we loop over each coarse grid point (I,J) and compute its value using the full-weighting stencil applied to the 
    //corresponding fine grid points.     
    for (int I = 0; I < n_coarse; I++) {
        for (int J = 0; J < n_coarse; J++) {
            int fi = 2*I + 1;    // fine grid index corresponding to coarse (I,J)
            int fj = 2*J + 1;    // (coarse point sits at odd fine indices)        

            //
            coarse[I*n_coarse + J] =
                (1.0/16.0) * ( F(fi-1, fj-1) + 2.0*F(fi-1, fj) + F(fi-1, fj+1) + 2.0*F(fi, fj-1) + 4.0*F(fi, fj) + 2.0*F(fi, fj+1)
                             + F(fi+1, fj-1) + 2.0*F(fi+1, fj) + F(fi+1, fj+1));
        }
    }

    #undef F
}

/* ------------------------------------------------------------------ */
/* Prolongation bilinear interpolation section                           */
/* ------------------------------------------------------------------ */

/*
 * This is our prolongation function, which uses bilinear interpolation to go from a coarse grid (n_coarse interior points)
 * to a fine grid (n_fine = 2*n_coarse + 1 interior points).
 *
 * We loop over fine grid points and classify them into four cases based
 * on whether their indices are odd or even:
 *
 *   (odd i,  odd j):  coincides with coarse point ((i-1)/2, (j-1)/2)
 *                     → direct injection
 *   (odd i,  even j): lies between two coarse points horizontally
 *                     → average of left and right coarse neighbours
 *   (even i, odd j):  lies between two coarse points vertically
 *                     → average of top and bottom coarse neighbours
 *   (even i, even j): lies at corner of four coarse points
 *                     → average of all four coarse neighbours
 */
void prolongate2d(int n_coarse, const double *coarse, int n_fine, double *fine)
{
    //We define this macro to help with accessing the coarse grid values with zero padding for out-of-bounds indices.
    #define C(ci, cj) \
        (((ci) >= 0 && (ci) < n_coarse && (cj) >= 0 && (cj) < n_coarse) \
         ? coarse[(ci)*n_coarse + (cj)] : 0.0)

    //Loop over fine grid points and apply the appropriate interpolation based on the parity of the indices.
    for (int i = 0; i < n_fine; i++) {
        for (int j = 0; j < n_fine; j++) {

            //We initialise this to store the interpolated value for fine point (i,j).
            double val;

            if (i % 2 == 1 && j % 2 == 1) {
                // Coincides with coarse point ((i-1)/2, (j-1)/2) 
                val = C((i-1)/2, (j-1)/2);

            } else if (i % 2 == 1 && j % 2 == 0) {
                // Between two coarse points horizontally
                val = 0.5 * (C((i-1)/2, j/2 - 1) + C((i-1)/2, j/2));

            } else if (i % 2 == 0 && j % 2 == 1) {
                // Between two coarse points vertically
                val = 0.5 * (C(i/2 - 1, (j-1)/2) + C(i/2, (j-1)/2));

            } else {
                // Corner of four coarse points
                val = 0.25 * (C(i/2 - 1, j/2 - 1) + C(i/2 - 1, j/2)
                            + C(i/2,     j/2 - 1) + C(i/2,     j/2));
            }

            //We add the interpolated value to the fine grid point (i,j).
            fine[i*n_fine + j] += val;
        }
    }

    #undef C
}

/* ------------------------------------------------------------------ */
/* V-cycle section                                                     */
/* ------------------------------------------------------------------ */

/*
 * This is our vcycle function, which implements a recursive V-cycle.
 * On entry, u is the current iterate (may be zero) and f is the right-hand side at this level.
 *
 * Strcture of the V-cycle:
 *   1. Pre-smooth:   smooth(nu1 steps)
 *   2. Residual:     r = f - A*u
 *   3. Restrict:     r_c = R * r
 *   4. Recurse:      vcycle on coarse grid with u_c = 0, f_c = r_c
 *   5. Prolongate:   u += P * u_c
 *   6. Post-smooth:  smooth(nu2 steps)
 */
void vcycle(int n_int, double h, double omega, int nu1, int nu2, SmootherFn smoother, double *u, const double *f)
{
    // 1x1 grid case, exact solve 
    if (n_int == 1) {
        u[0] = f[0] * h * h / 4.0;
        return;
    }

    //Total number of interior grid points at this level
    int N = n_int * n_int;

    // 1. Pre-smoothing 
    smoother(n_int, h, omega, nu1, u, f);

    // 2. Compute residual r = f - A*u
    double *r = malloc(N * sizeof(double));
    matvec_poisson2d(n_int, h, u, r);          // r = A*u   
    for (int k = 0; k < N; k++)
        r[k] = f[k] - r[k];                   // r = f - Au

    // 3. Restrict residual to coarse grid
    int n_coarse = (n_int - 1) / 2;
    int N_coarse = n_coarse * n_coarse;
    double h_coarse = 1.0 / (n_coarse + 1);

    double *r_coarse = malloc(N_coarse * sizeof(double));
    double *u_coarse = calloc(N_coarse, sizeof(double));  /* zero initial guess */

    restrict2d(n_int, r, r_coarse);

    // 4. Recursive V-cycle on coarse grid
    vcycle(n_coarse, h_coarse, omega, nu1, nu2, smoother, u_coarse, r_coarse);

    // 5. Prolongate coarse correction and add to fine grid solution
    prolongate2d(n_coarse, u_coarse, n_int, u);

    // 6. Post-smoothing
    smoother(n_int, h, omega, nu2, u, f);

    //Free dyncamically allocated memory for the residual and coarse grid vectors.
    free(r);
    free(r_coarse);
    free(u_coarse);
}