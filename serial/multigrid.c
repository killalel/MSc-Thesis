#include <stdlib.h>
#include <string.h>
#include "multigrid.h"
#include "linalg.h"

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
void wj_smooth(int n_int, double h, double omega, int nu,
            double *u, const double *f)
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
void vcycle(int n_int, double h, double omega, int nu1, int nu2, double *u, const double *f)
{
    // 1x1 grid case, exact solve 
    if (n_int == 1) {
        u[0] = f[0] * h * h / 4.0;
        return;
    }

    //Total number of interior grid points at this level
    int N = n_int * n_int;

    // 1. Pre-smoothing 
    wj_smooth(n_int, h, omega, nu1, u, f);

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
    vcycle(n_coarse, h_coarse, omega, nu1, nu2, u_coarse, r_coarse);

    // 5. Prolongate coarse correction and add to fine grid solution
    prolongate2d(n_coarse, u_coarse, n_int, u);

    // 6. Post-smoothing
    wj_smooth(n_int, h, omega, nu2, u, f);

    //Free dyncamically allocated memory for the residual and coarse grid vectors.
    free(r);
    free(r_coarse);
    free(u_coarse);
}