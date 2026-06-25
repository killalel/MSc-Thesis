#include <math.h>
#include <stdlib.h>
#include "poisson2d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * poisson2d_driven_cavity.c
 *
 * This file implements the 2D Poisson problem with a non-zero RHS motivated
 * by the pressure Poisson equation in CFD applications (see Idomura et al.).
 *
 * Problem:
 *   -Delta u = f    on (0,1)^2
 *   u = 0           on the boundary (homogeneous Dirichlet)
 *
 * Analytic solution:
 *   u(x,y) = sin(2*pi*x) * sin(2*pi*y)
 *
 * Source term (from -Delta u):
 *   f(x,y) = 8*pi^2 * sin(2*pi*x) * sin(2*pi*y)
 *
 * Since all BCs are zero, the RHS is purely from the source term —
 * no boundary contributions need to be added.
 */

//All four boundary conditions are zero (homogeneous Dirichlet).
static double bc_left(double y)   { (void)y; return 0.0; }
static double bc_right(double y)  { (void)y; return 0.0; }
static double bc_bottom(double x) { (void)x; return 0.0; }
static double bc_top(double x)    { (void)x; return 0.0; }

//This function computes the analytic solution at a given interior grid point (i,j).
double analytic_solution(int i, int j, int n_int)
{
    double h = 1.0 / (n_int + 1);
    double x = i * h;
    double y = j * h;
    return sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
}

//This is our function for building the right hand side vector b.
void build_rhs(int n_int, double *b)
{
    //This is just the grid spacing.
    double h     = 1.0 / (n_int + 1);
    //This is just 1/h^2
    double h2inv = 1.0 / (h * h);

    //Loop over all interior grid points and set b[k] = f(x,y).
    //Since the BCs are all zero there are no boundary contributions to add.
    for (int i = 1; i <= n_int; i++) {
        for (int j = 1; j <= n_int; j++) {
            int k = (i-1)*n_int + (j-1);   // 0-based flat index
            double x = i * h;   // x coordinate of this grid point
            double y = j * h;   // y coordinate of this grid point

            // Source term f = 8*pi^2 * sin(2*pi*x) * sin(2*pi*y)
            b[k] = 8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);

            // BC contributions — all zero so nothing to add, but we keep
            // the structure here for clarity and future reference.
            if (i == 1)     b[k] += h2inv * bc_left(y);
            if (i == n_int) b[k] += h2inv * bc_right(y);
            if (j == 1)     b[k] += h2inv * bc_bottom(x);
            if (j == n_int) b[k] += h2inv * bc_top(x);
        }
    }
}

//This function computes the maximum error of the computed solution x against the analytic solution at the interior grid points.
double max_error(int n_int, const double *x)
{
    double err = 0.0;
    for (int i = 1; i <= n_int; i++) {
        for (int j = 1; j <= n_int; j++) {
            int k = (i-1)*n_int + (j-1);   // 0-based flat index
            double diff = fabs(x[k] - analytic_solution(i, j, n_int)); //The difference between the computed solution and the analytic solution at this grid point.
            if (diff > err) err = diff; //Update the maximum error if this point has a larger error than the current maximum.
        }
    }
    //Returns the error
    return err;
}