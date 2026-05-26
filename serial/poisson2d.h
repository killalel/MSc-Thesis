#ifndef POISSON2D_H
#define POISSON2D_H

/*
 * poisson2d.h
 *
 * This is the header file for our 2D Poisson problem. It contains function prototypes for building the right-hand side vector from the
 * boundary conditions, computing the maximum error against the analytic solution, and the analytic solution itself. 
 * 
 * Problem setup for the 2D Poisson BVP:
 *   -Delta u = f    on (0,1)^2
 * with the non-homogeneous Dirichlet boundary conditions from
 * the HPC SWII assignment, whose analytic solution is:
 *   u(x,y) = y / ((1+x)^2 + y^2)
 * Interior grid: n_int x n_int points, mesh spacing h = 1/(n_int+1).
 * Flat index convention: k = i*n_int + j  (i = x-dir, j = y-dir).
 * 
 * See the .c file for more detailed comments on the implementation of these functions.
 */


double analytic_solution(int i, int j, int n_int);


void build_rhs(int n_int, double *b);


double max_error(int n_int, const double *x);

#endif /* POISSON2D_H */