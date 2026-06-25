#ifndef LANCZOS_H
#define LANCZOS_H

/*
 * lanczos.h
 *
 * This is the header file for our Lanczos implementation. It defines the interface for computing extremal eigenvalues of the 2D Poisson operator 
 * using the Lanczos algorithm with partial reorthogonalization and a block Jacobi preconditioner.
 * 
 * The main function is lanczos_eigenvalues, which takes in the tridiagonal and subdiagonal arrays from the Lanczos iteration, as well as the number of interior points and mesh spacing, and computes estimates for the smallest and largest eigenvalues of the preconditioned operator.
 * 
 * More detailed comments can be found in the corresponding lanczos.c file, which contains the implementation of this function.
 */


void lanczos_eigenvalues(int n_int, double h, const double *td, const double *tl, int k, double *lam_min, double *lam_max);

#endif /* LANCZOS_H */