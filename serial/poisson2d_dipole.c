#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "poisson2d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * poisson2d_dipole.c
 *
 * This file implements the 2D Poisson problem with a Gaussian dipole source term,
 * motivated by electrostatic or heat source/sink problems.
 *
 * Problem:
 *   -Delta u = f    on (0,1)^2
 *   u = 0           on the boundary (homogeneous Dirichlet)
 *
 * Source term:
 *   f(x,y) = exp(-100((x-0.3)^2 + (y-0.5)^2))
 *           - exp(-100((x-0.7)^2 + (y-0.5)^2))
 *
 * This is a Gaussian dipole — a positive source at (0.3, 0.5) and a negative
 * sink at (0.7, 0.5). It has no closed-form analytic solution, so correctness
 * is verified via grid convergence study rather than pointwise error.
 */

//All four boundary conditions are zero (homogeneous Dirichlet).
static double bc_left(double y)   { (void)y; return 0.0; }
static double bc_right(double y)  { (void)y; return 0.0; }
static double bc_bottom(double x) { (void)x; return 0.0; }
static double bc_top(double x)    { (void)x; return 0.0; }

//No analytic solution exists for this problem — returns 0.
//Correctness is verified via grid convergence study instead.
double analytic_solution(int i, int j, int n_int)
{
    (void)i; (void)j; (void)n_int;
    return 0.0;
}

//This is our function for building the right hand side vector b.
void build_rhs(int n_int, double *b)
{
    //This is just the grid spacing.
    double h     = 1.0 / (n_int + 1);
    //This is just 1/h^2
    double h2inv = 1.0 / (h * h);

    //Loop over all interior grid points and set b[k] = f(x,y).
    for (int i = 1; i <= n_int; i++) {
        for (int j = 1; j <= n_int; j++) {
            int k = (i-1)*n_int + (j-1);   // 0-based flat index
            double x = i * h;   // x coordinate of this grid point
            double y = j * h;   // y coordinate of this grid point

            //Gaussian dipole source term: positive source at (0.3, 0.5),
            //negative sink at (0.7, 0.5).
            double r1 = (x - 0.3)*(x - 0.3) + (y - 0.5)*(y - 0.5);
            double r2 = (x - 0.7)*(x - 0.7) + (y - 0.5)*(y - 0.5);
            b[k] = exp(-100.0 * r1) - exp(-100.0 * r2);

            //BC contributions — all zero so nothing to add.
            if (i == 1)     b[k] += h2inv * bc_left(y);
            if (i == n_int) b[k] += h2inv * bc_right(y);
            if (j == 1)     b[k] += h2inv * bc_bottom(x);
            if (j == n_int) b[k] += h2inv * bc_top(x);
        }
    }
}

//No analytic solution is available for this problem.
//Prints a warning and returns -1. Use the PCG residual history and
//grid convergence study to verify correctness instead.
double max_error(int n_int, const double *x)
{
    (void)n_int; (void)x;
    printf("Note: no analytic solution available for dipole problem.\n");
    printf("Verify correctness via grid convergence study.\n\n");
    return -1.0;
}