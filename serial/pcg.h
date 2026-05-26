#ifndef PCG_H
#define PCG_H
 
#include "precond.h"

//This is the header file for our implementation of the Preconditioned Conjugate Gradient (PCG) solver for the 2D Poisson problem.
int pcg_solve(int n_int, const double *b, double *x, double tol, int max_iter,PrecondApplyFn precond_apply, void *precond_ctx, double *rel_res);
 
#endif /* PCG_H */
 