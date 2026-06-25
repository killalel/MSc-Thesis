#ifndef PCG_H
#define PCG_H

#include "decomp.h"
#include "precond.h"

// Distributed PCG solver for the 2D Poisson system.
// Algorithm follows Saad Algorithm 9.1 (Preconditioned CG).

int pcg_solve(const double *b, double *x, double tol, int max_iter, PrecondApplyFn precond_apply, void *precond_ctx, double *rel_res, const Decomp *d);

#endif // PCG_H