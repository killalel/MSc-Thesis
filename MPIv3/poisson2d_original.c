#include <math.h>
#include "poisson2d.h"

// Boundary condition functions which are the same as the serial version.
static double bc_left(double y) { return y / (1.0 + y * y); }
static double bc_right(double y) { return y / (4.0 + y * y); }
static double bc_bottom(double x) { (void)x; return 0.0; }
static double bc_top(double x) { return 1.0 / ((1.0 + x) * (1.0 + x) + 1.0); }

//This is just the analytic solution.
double analytic_solution(int gi, int gj, int n_global)
{
    double h = 1.0 / (n_global + 1);
    double x = gi * h;
    double y = gj * h;
    return y / ((1.0 + x) * (1.0 + x) + y * y);
}

//This builds the RHS
void build_rhs(double *b, const Decomp *d)
{
    int nx = d->n_local[1];
    int ny = d->n_local[0];
    double h = d->h;
    double h2inv = 1.0 / (h * h);

    // Zero out first because the source term f = 0 everywhere in the interior.
    for (int k = 0; k < nx * ny; k++)
        b[k] = 0.0;

    // For each local interior point, check if any of its four neighbours
    // falls on the physical boundary. If so, add the known BC value * h2inv.
    // Global 1-based indices: gi = offsets[1] + j + 1, gj = offsets[0] + i + 1
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int k  = i * nx + j;
            int gi = d->offsets[1] + j + 1;   // global x-index (1-based)
            int gj = d->offsets[0] + i + 1;   // global y-index (1-based)
            double x = gi * h;
            double y = gj * h;

            // Left physical boundary (gi == 1 and no left neighbour)
            if (gi == 1 && d->neighbours[0] == MPI_PROC_NULL)
                b[k] += h2inv * bc_left(y);

            // Right physical boundary (gi == n_global and no right neighbour)
            if (gi == d->n_global && d->neighbours[1] == MPI_PROC_NULL)
                b[k] += h2inv * bc_right(y);

            // Bottom physical boundary (gj == 1 and no bottom neighbour)
            if (gj == 1 && d->neighbours[2] == MPI_PROC_NULL)
                b[k] += h2inv * bc_bottom(x);

            // Top physical boundary (gj == n_global and no top neighbour)
            if (gj == d->n_global && d->neighbours[3] == MPI_PROC_NULL)
                b[k] += h2inv * bc_top(x);
        }
    }
}

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