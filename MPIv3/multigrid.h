#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "decomp.h"

// Smoother function pointer type. Same modular pattern as the serial version.
// The Decomp* is needed so smoothers can call the distributed matvec.
typedef void (*SmootherFn)(double *u, const double *f, double omega, int nu, const Decomp *d);

// Weighted Jacobi smoother.
// Applies nu sweeps of: u = u + omega * D^{-1} * (f - A*u)
// The distributed matvec handles the halo exchange internally.
void wj_smooth(double *u, const double *f, double omega, int nu, const Decomp *d);

void gs_smooth(double *u, const double *f, double omega, int nu, const Decomp *d);
void rb_smooth(double *u, const double *f, double omega, int nu, const Decomp *d);
void sor_smooth(double *u, const double *f, double omega, int nu, const Decomp *d);

// Full-weighting restriction: fine grid residual -> coarse grid.
// Does a halo exchange on r_fine before applying the 9-point stencil.
void restrict2d(const double *r_fine, const Decomp *df, double *r_coarse, const Decomp *dc);

// Bilinear prolongation: coarse correction -> fine grid.
// Does a halo exchange on u_coarse before interpolating.
// The correction is ADDED to u_fine.
void prolongate2d(const double *u_coarse, const Decomp *dc, double *u_fine, const Decomp *df);

// One V-cycle of the distributed multigrid preconditioner.
// Uses the redundant coarse grid solve strategy at the coarsest level.
void vcycle(int level, int n_levels, Decomp **decomps, double **work_u, double **work_f, double **work_r, double *u, const double *f, double omega,
            int nu1, int nu2, SmootherFn smoother);

#endif // MULTIGRID_H
