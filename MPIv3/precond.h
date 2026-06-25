#ifndef PRECOND_H
#define PRECOND_H

#include "decomp.h"

// Generic preconditioner function pointer type.
//Parameters:
// z [out] result of solving M*z = r
// r [in]  right-hand side
// ctx [in] preconditioner context cast to void*
typedef void (*PrecondApplyFn)(double *z, const double *r, void *ctx);

// Identity preconditioner 
typedef struct { int n; } IdentityCtx;
IdentityCtx *identity_setup(int n);
void identity_apply(double *z, const double *r, void *ctx);
void identity_free(IdentityCtx *ctx);

// Block Jacobi preconditioner.
// The block size is d->n_local[1] — the number of local x-direction points.
// Each rank factors one set of Thomas factors and applies them to all
// d->n_local[0] row-blocks independently, exactly as in the serial version.
typedef struct {
    int nx; // local points in x-direction (block size)
    int ny; // local points in y-direction (number of blocks)
    double h; // mesh spacing
    double *l; // Thomas multipliers
    double *d; // modified diagonal after elimination
} BlockJacobiCtx;

BlockJacobiCtx *block_jacobi_setup(const Decomp *d);
void block_jacobi_apply(double *z, const double *r, void *ctx);
void block_jacobi_free(BlockJacobiCtx *ctx);

// Multigrid preconditioner.
// Stores the full level hierarchy and pre-allocated work arrays.
// The redundant coarse grid strategy is used at the coarsest level —
// all ranks assemble the full coarse residual via MPI_Allreduce and
// solve independently.
#include "multigrid.h"

typedef struct {
    int n_levels;     // total number of levels
    Decomp **decomps;      // decomps[0] = fine, decomps[n_levels-1] = coarsest
    double **work_u;       // pre-allocated solution vectors for each level
    double **work_f;       // pre-allocated RHS vectors for each level
    double **work_r;       // pre-allocated residual vectors for each level
    int nu1;          // pre-smoothing steps
    int nu2;          // post-smoothing steps
    double omega;        // Jacobi damping weight
    SmootherFn smoother;     // smoother function pointer
} MultigridCtx;

MultigridCtx *multigrid_setup(const Decomp *d, int nu1, int nu2,
                              double omega, SmootherFn smoother);
void multigrid_apply(double *z, const double *r, void *ctx);
void multigrid_free(MultigridCtx *ctx);

#endif // PRECOND_H
