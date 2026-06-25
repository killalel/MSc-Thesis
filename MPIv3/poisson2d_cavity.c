#include <math.h>
#include "poisson2d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//This file implements the 2D Poisson problem with a non-zero RHS motivated
//by the pressure Poisson equation in CFD applications (see Idomura et al.).


// -Delta u = f on (0,1)^2, u = 0 on the boundary (homogeneous Dirichlet).
// Analytic solution: u(x,y) = sin(2*pi*x) * sin(2*pi*y)
// Source term: f(x,y) = 8*pi^2 * sin(2*pi*x) * sin(2*pi*y)
// All BCs are zero, so build_rhs has no boundary contributions.

double analytic_solution(int gi, int gj, int n_global)
{
    double h = 1.0 / (n_global + 1);
    double x = gi * h;
    double y = gj * h;
    return sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
}

//This is our function for building the right hand side vector b.
void build_rhs(double *b, const Decomp *d)
{
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double h = d->h;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int k  = i * nx + j;
            int gi = d->offsets[1] + j + 1;
            int gj = d->offsets[0] + i + 1;
            double x = gi * h;
            double y = gj * h;

            b[k] = 8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
        }
    }
}

//This function calculates the maximum error between our version and the analytic solution.
double max_error(const double *x, const Decomp *d)
{
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double local_err = 0.0;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int gi = d->offsets[1] + j + 1;
            int gj = d->offsets[0] + i + 1;
            double diff = fabs(x[i * nx + j] - analytic_solution(gi, gj, d->n_global));
            if (diff > local_err) local_err = diff;
        }
    }

    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, d->cart_comm);
    return global_err;
}