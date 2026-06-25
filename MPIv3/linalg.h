#ifndef LINALG_H
#define LINALG_H

#include "decomp.h"

// Basic vector operations. All vectors have length d->n_local[0] * d->n_local[1]
// and represent the interior points owned by this rank, stored in row-major order
// with NO halo. Halo storage is managed internally by matvec_poisson2d.

// y = y + alpha * x
void vec_axpy(int n, double alpha, const double *x, double *y);

// y = x
void vec_copy(int n, const double *x, double *y);

// x = 0
void vec_zero(int n, double *x);

// Returns the global dot product x^T y via MPI_Allreduce.
double vec_dot(int n, const double *x, const double *y, const Decomp *d);

// Returns the global 2-norm ||x||_2 via vec_dot.
double vec_norm2(int n, const double *x, const Decomp *d);

// Distributed matvec: y = A*x where A is the 2D Poisson matrix.
// Performs a non-blocking halo exchange on x before applying the stencil.
// x and y are local vectors of length d->n_local[0] * d->n_local[1].
void matvec_poisson2d(const double *x, double *y, const Decomp *d);

void halo_exchange(const double *x, int nx, int ny, double *recv_left, double *recv_right, double *recv_bottom, double *recv_top, const Decomp *d);
#endif // LINALG_H
