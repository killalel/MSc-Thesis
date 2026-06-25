#ifndef POISSON2D_H
#define POISSON2D_H

#include "decomp.h"

// Analytic solution at global interior point (gi, gj), 1-based.
// u(x,y) = y / ((1+x)^2 + y^2)
double analytic_solution(int gi, int gj, int n_global);

// Builds the local portion of the RHS vector b.
// b has length d->n_local[0] * d->n_local[1].
// Each rank only fills in the entries for its own subdomain, including
// boundary contributions from the non-homogeneous Dirichlet BCs.
void build_rhs(double *b, const Decomp *d);

// Computes the global max pointwise error between the local solution x
// and the analytic solution. Uses MPI_Allreduce to find the global max.
double max_error(const double *x, const Decomp *d);

#endif // POISSON2D_H