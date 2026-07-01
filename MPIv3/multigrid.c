#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "multigrid.h"
#include "linalg.h"

// wj_smooth
//
// Weighted Jacobi smoother for A*u = f on the local subdomain.
// The distributed matvec handles the halo exchange so this function
// is almost identical to the serial version — just pass d through.
void wj_smooth(double *u, const double *f, double omega, int nu, const Decomp *d)
{
    int N = d->n_local[0] * d->n_local[1];
    double diag  = 4.0 / (d->h * d->h);
    double scale = omega / diag;

    double *Au = malloc(N * sizeof(double));

    for (int sweep = 0; sweep < nu; sweep++) {
        matvec_poisson2d(u, Au, d);
        for (int k = 0; k < N; k++)
            u[k] += scale * (f[k] - Au[k]);
    }

    free(Au);
}

//Symmetric Gauss-Seidel smoother. Each rank does GS locally using
//stale halo values — effectively "local GS / global Jacobi". A halo
//exchange before each sweep direction refreshes the boundary data.
void gs_smooth(double *u, const double *f, double omega, int nu, const Decomp *d)
{
    (void)omega;
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double h2 = d->h * d->h;

    double *hl = malloc(ny * sizeof(double));
    double *hr = malloc(ny * sizeof(double));
    double *hb = malloc(nx * sizeof(double));
    double *ht = malloc(nx * sizeof(double));

    for (int sweep = 0; sweep < nu; sweep++) {
        //forward pass
        halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                double nb = 0.0;
                if (j > 0)          nb += u[i * nx + (j - 1)];
                else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                if (i > 0)          nb += u[(i - 1) * nx + j];
                else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                u[i * nx + j] = (h2 * f[i * nx + j] + nb) / 4.0;
            }
        }

        //backward pass
        halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
        for (int i = ny - 1; i >= 0; i--) {
            for (int j = nx - 1; j >= 0; j--) {
                double nb = 0.0;
                if (j > 0)          nb += u[i * nx + (j - 1)];
                else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                if (i > 0)          nb += u[(i - 1) * nx + j];
                else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                u[i * nx + j] = (h2 * f[i * nx + j] + nb) / 4.0;
            }
        }
    }

    free(hl); free(hr); free(hb); free(ht);
}


//Symmetric Red-Black Gauss-Seidel smoother. Points are coloured by
//global index parity: red if (gi+gj) even, black if odd. Within each
//colour all updates are independent, so this parallelises cleanly —
//only the halo exchange between colour passes is needed.
void rb_smooth(double *u, const double *f, double omega, int nu, const Decomp *d)
{
    (void)omega;
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double h2 = d->h * d->h;

    double *hl = malloc(ny * sizeof(double));
    double *hr = malloc(ny * sizeof(double));
    double *hb = malloc(nx * sizeof(double));
    double *ht = malloc(nx * sizeof(double));

    for (int sweep = 0; sweep < nu; sweep++) {
        //forward: red then black
        for (int colour = 0; colour <= 1; colour++) {
            halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
            for (int i = 0; i < ny; i++) {
                int gi = d->offsets[0] + i;
                for (int j = 0; j < nx; j++) {
                    int gj = d->offsets[1] + j;
                    if ((gi + gj) % 2 != colour) continue;
                    double nb = 0.0;
                    if (j > 0)          nb += u[i * nx + (j - 1)];
                    else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                    if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                    else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                    if (i > 0)          nb += u[(i - 1) * nx + j];
                    else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                    if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                    else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                    u[i * nx + j] = (h2 * f[i * nx + j] + nb) / 4.0;
                }
            }
        }

        //backward: black then red
        for (int colour = 1; colour >= 0; colour--) {
            halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
            for (int i = ny - 1; i >= 0; i--) {
                int gi = d->offsets[0] + i;
                for (int j = nx - 1; j >= 0; j--) {
                    int gj = d->offsets[1] + j;
                    if ((gi + gj) % 2 != colour) continue;
                    double nb = 0.0;
                    if (j > 0)          nb += u[i * nx + (j - 1)];
                    else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                    if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                    else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                    if (i > 0)          nb += u[(i - 1) * nx + j];
                    else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                    if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                    else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                    u[i * nx + j] = (h2 * f[i * nx + j] + nb) / 4.0;
                }
            }
        }
    }

    free(hl); free(hr); free(hb); free(ht);
}


//Symmetric SOR smoother. Same structure as GS but blends the
//Gauss-Seidel update with the old value: u = (1-omega)*u_old + omega*u_gs.
void sor_smooth(double *u, const double *f, double omega, int nu, const Decomp *d)
{
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double h2 = d->h * d->h;

    double *hl = malloc(ny * sizeof(double));
    double *hr = malloc(ny * sizeof(double));
    double *hb = malloc(nx * sizeof(double));
    double *ht = malloc(nx * sizeof(double));

    for (int sweep = 0; sweep < nu; sweep++) {
        //forward pass
        halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                double nb = 0.0;
                if (j > 0)          nb += u[i * nx + (j - 1)];
                else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                if (i > 0)          nb += u[(i - 1) * nx + j];
                else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                double gs = (h2 * f[i * nx + j] + nb) / 4.0;
                u[i * nx + j] = (1.0 - omega) * u[i * nx + j] + omega * gs;
            }
        }

        //backward pass
        halo_exchange(u, nx, ny, hl, hr, hb, ht, d);
        for (int i = ny - 1; i >= 0; i--) {
            for (int j = nx - 1; j >= 0; j--) {
                double nb = 0.0;
                if (j > 0)          nb += u[i * nx + (j - 1)];
                else if (d->neighbours[0] != MPI_PROC_NULL) nb += hl[i];
                if (j < nx - 1)     nb += u[i * nx + (j + 1)];
                else if (d->neighbours[1] != MPI_PROC_NULL) nb += hr[i];
                if (i > 0)          nb += u[(i - 1) * nx + j];
                else if (d->neighbours[2] != MPI_PROC_NULL) nb += hb[j];
                if (i < ny - 1)     nb += u[(i + 1) * nx + j];
                else if (d->neighbours[3] != MPI_PROC_NULL) nb += ht[j];
                double gs = (h2 * f[i * nx + j] + nb) / 4.0;
                u[i * nx + j] = (1.0 - omega) * u[i * nx + j] + omega * gs;
            }
        }
    }

    free(hl); free(hr); free(hb); free(ht);
}


// restrict2d
//
// Full-weighting restriction from fine grid (df) to coarse grid (dc).
// Before applying the 9-point stencil we need halo values from neighbouring
// ranks on the fine grid, so we do a halo exchange on r_fine first.
// The macro F(fi,fj) accesses the fine grid with halo fallback.
void restrict2d(const double *r_fine, const Decomp *df, double *r_coarse, const Decomp *dc)
{
    int fnx = df->n_local[1];
    int fny = df->n_local[0];
    int cnx = dc->n_local[1];
    int cny = dc->n_local[0];

    //Halo exchange on fine residual.
    double *hl = malloc(fny * sizeof(double));
    double *hr = malloc(fny * sizeof(double));
    double *hb = malloc(fnx * sizeof(double));
    double *ht = malloc(fnx * sizeof(double));
    halo_exchange(r_fine, fnx, fny, hl, hr, hb, ht, df);

    //Helper macro: fine grid value at local index (fi, fj), using halo values
    //for out-of-bounds indices and 0 for physical boundary.
    #define F(fi, fj) ( \
        ((fi) >= 0 && (fi) < fny && (fj) >= 0 && (fj) < fnx) \
            ? r_fine[(fi)*fnx + (fj)] \
        : ((fi) == -1 && (fj) >= 0 && (fj) < fnx) \
            ? hb[(fj)] \
        : ((fi) == fny && (fj) >= 0 && (fj) < fnx) \
            ? ht[(fj)] \
        : ((fj) == -1 && (fi) >= 0 && (fi) < fny) \
            ? hl[(fi)] \
        : ((fj) == fnx && (fi) >= 0 && (fi) < fny) \
            ? hr[(fi)] \
        : 0.0 )

    //Each coarse point (I,J) in local indices corresponds to fine point
    //(2I+1, 2J+1) in local indices. Apply the 9-point full-weighting stencil.
    for (int I = 0; I < cny; I++) {
        for (int J = 0; J < cnx; J++) {
            int fi = 2*I + 1;
            int fj = 2*J + 1;
            r_coarse[I*cnx + J] = (1.0/16.0) * ( F(fi-1, fj-1) + 2.0*F(fi-1, fj) + F(fi-1, fj+1)  + 2.0*F(fi,   fj-1) + 4.0*F(fi,   fj) + 
                                   2.0*F(fi, fj+1) + F(fi+1, fj-1) + 2.0*F(fi+1, fj) + F(fi+1, fj+1));
        }
    }

    #undef F
    free(hl); 
    free(hr); 
    free(hb); 
    free(ht);
}

// prolongate2d
//
// Bilinear interpolation from coarse grid (dc) to fine grid (df).
// We need halo values from coarse grid neighbours before interpolating,
// so we do a halo exchange on u_coarse first.
// The correction is ADDED to u_fine.
//
// Fine point classification uses global indices to determine parity,
// because local index parity depends on where the subdomain sits in
// the global grid.
void prolongate2d(const double *u_coarse, const Decomp *dc, double *u_fine, const Decomp *df)
{
    int cnx = dc->n_local[1];
    int cny = dc->n_local[0];
    int fnx = df->n_local[1];
    int fny = df->n_local[0];

    //Halo exchange on coarse correction.
    double *hl = malloc(cny * sizeof(double));
    double *hr = malloc(cny * sizeof(double));
    double *hb = malloc(cnx * sizeof(double));
    double *ht = malloc(cnx * sizeof(double));
    halo_exchange(u_coarse, cnx, cny, hl, hr, hb, ht, dc);

    //Helper macro: coarse value at local index (ci, cj), using halo values
    //for out-of-bounds indices and 0 for physical boundary.
    #define C(ci, cj) ( \
        ((ci) >= 0 && (ci) < cny && (cj) >= 0 && (cj) < cnx) \
            ? u_coarse[(ci)*cnx + (cj)] \
        : ((ci) == -1 && (cj) >= 0 && (cj) < cnx) \
            ? hb[(cj)] \
        : ((ci) == cny && (cj) >= 0 && (cj) < cnx) \
            ? ht[(cj)] \
        : ((cj) == -1 && (ci) >= 0 && (ci) < cny) \
            ? hl[(ci)] \
        : ((cj) == cnx && (ci) >= 0 && (ci) < cny) \
            ? hr[(ci)] \
        : 0.0 )

    for (int i = 0; i < fny; i++) {
        for (int j = 0; j < fnx; j++) {
            //Global fine indices, 1-based to match the serial convention
            //where interior points run from 1 to n_global.
            int gi = df->offsets[0] + i;
            int gj = df->offsets[1] + j;

            //Coarse local indices — coarse points sit at odd 1-based fine indices.
            int co0 = dc->offsets[0];
            int co1 = dc->offsets[1];

            double val;
            if (gi % 2 == 1 && gj % 2 == 1) {
                val = C((gi-1)/2 - co0, (gj-1)/2 - co1);
            } else if (gi % 2 == 1 && gj % 2 == 0) {
                val = 0.5 * (C((gi-1)/2 - co0, gj/2 - 1 - co1) + C((gi-1)/2 - co0, gj/2     - co1));
            } else if (gi % 2 == 0 && gj % 2 == 1) {
                val = 0.5 * (C(gi/2 - 1 - co0, (gj-1)/2 - co1) + C(gi/2     - co0, (gj-1)/2 - co1));
            } else {
                val = 0.25 * (C(gi/2 - 1 - co0, gj/2 - 1 - co1) + C(gi/2 - 1 - co0, gj/2     - co1) + C(gi/2     - co0, gj/2 - 1 - co1)
                     + C(gi/2     - co0, gj/2     - co1));
            }

            u_fine[i * fnx + j] += val;
        }
    }

    #undef C
    free(hl); 
    free(hr); 
    free(hb); 
    free(ht);
}

// vcycle
//
// Distributed recursive V-cycle. At the coarsest level all ranks assemble
// the full residual via MPI_Allreduce and solve exactly in serial — the
// redundant coarse grid strategy.
void vcycle(int level, int n_levels, Decomp **decomps, double **work_u, double **work_f, double **work_r, double *u, const double *f, double omega, 
            int nu1, int nu2, SmootherFn smoother)
{
    Decomp *d  = decomps[level];
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    int N = nx * ny;

    //Base case: coarsest level — redundant exact solve.
    //Each rank assembles the full coarse f via MPI_Allreduce, then all ranks
    //solve independently. Works because the coarsest grid is tiny and cheap.
    if (level == n_levels - 1) {
        int n_global = d->n_global;
        int N_global = n_global * n_global;

        double *f_global = calloc(N_global, sizeof(double));
        double *u_global = calloc(N_global, sizeof(double));

        //Pack local f into the global array at the correct global positions.
        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++)
                f_global[(d->offsets[0] + i) * n_global + (d->offsets[1] + j)] = f[i * nx + j];

        //Sum across all ranks so every rank has the full f_global.
        MPI_Allreduce(MPI_IN_PLACE, f_global, N_global, MPI_DOUBLE, MPI_SUM, d->cart_comm);

        //Serial weighted Jacobi on the full coarse grid until residual is small.
        double h = d->h;
        double h2 = h * h;
        double diag = 4.0 / h2;
        double *Au = malloc(N_global * sizeof(double));

        for (int iter = 0; iter < 500; iter++) {
            for (int i = 0; i < n_global; i++) {
                for (int j = 0; j < n_global; j++) {
                    int k = i * n_global + j;
                    double v = 4.0 * u_global[k];
                    if (i > 0) v -= u_global[(i-1)*n_global + j];
                    if (i < n_global - 1) v -= u_global[(i+1)*n_global + j];
                    if (j > 0) v -= u_global[i*n_global + (j-1)];
                    if (j < n_global - 1) v -= u_global[i*n_global + (j+1)];
                    Au[k] = v / h2;
                }
            }
            double res = 0.0;
            for (int k = 0; k < N_global; k++) {
                double r = f_global[k] - Au[k];
                u_global[k] += (omega / diag) * r;
                res += r * r;
            }
            if (sqrt(res) < 1e-12) break;
        }
        free(Au);

        //Unpack local portion of u_global back into u.
        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++)
                u[i * nx + j] = u_global[(d->offsets[0] + i) * n_global + (d->offsets[1] + j)];

        free(f_global);
        free(u_global);
        return;
    }

    //1. Pre-smoothing.
    smoother(u, f, omega, nu1, d);

    //2. Compute residual r = f - A*u.
    double *r = work_r[level];
    matvec_poisson2d(u, r, d);
    for (int k = 0; k < N; k++)
        r[k] = f[k] - r[k];

    //3. Restrict residual to coarse grid.
    Decomp *dc  = decomps[level + 1];
    int N_c = dc->n_local[0] * dc->n_local[1];
    double *f_c = work_f[level + 1];
    double *u_c = work_u[level + 1];

    restrict2d(r, d, f_c, dc);

    //4. Zero initial guess on coarse grid.
    memset(u_c, 0, N_c * sizeof(double));

    //5. Recursive V-cycle on coarse grid.
    vcycle(level + 1, n_levels, decomps, work_u, work_f, work_r, u_c, f_c, omega, nu1, nu2, smoother);

    //6. Prolongate correction back and add to fine grid solution.
    prolongate2d(u_c, dc, u, d);

    //7. Post-smoothing.
    smoother(u, f, omega, nu2, d);
}
