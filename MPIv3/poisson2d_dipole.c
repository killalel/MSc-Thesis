#include <math.h>
#include <stdio.h>
#include "poisson2d.h"

// -Delta u = f on (0,1)^2, u = 0 on the boundary (homogeneous Dirichlet).
// Source term: Gaussian dipole — positive source at (0.3,0.5), negative
// sink at (0.7,0.5). No analytic solution; verify via grid convergence.

//This is just the analytic solution function, which returns 0 since no analytic solution exists for this problem.
double analytic_solution(int gi, int gj, int n_global)
{
    (void)gi; 
    (void)gj; 
    (void)n_global;
    return 0.0;
}

//This builds the RHS.
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

            double r1 = (x - 0.3)*(x - 0.3) + (y - 0.5)*(y - 0.5);
            double r2 = (x - 0.7)*(x - 0.7) + (y - 0.5)*(y - 0.5);
            b[k] = exp(-100.0 * r1) - exp(-100.0 * r2);
        }
    }
}

//This is for the max error, but since no analytic solution exists, it prints a warning and returns -1.
double max_error(const double *x, const Decomp *d)
{
    (void)x;
    if (d->rank == 0) {
        printf("Note: no analytic solution available for dipole problem.\n");
        printf("Verify correctness via grid convergence study.\n\n");
    }
    return -1.0;
}