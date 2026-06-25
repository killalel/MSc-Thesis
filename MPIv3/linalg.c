#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "linalg.h"

//y = y + alpha * x
void vec_axpy(int n, double alpha, const double *x, double *y)
{
    for (int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

//y = x
void vec_copy(int n, const double *x, double *y)
{
    memcpy(y, x, n * sizeof(double));
}

//x = 0
void vec_zero(int n, double *x)
{
    memset(x, 0, n * sizeof(double));
}

//Returns the global dot product x^T y via MPI_Allreduce.
double vec_dot(int n, const double *x, const double *y, const Decomp *d)
{
    // Local partial dot product first, then sum across all ranks.
    double local = 0.0;
    for (int i = 0; i < n; i++)
        local += x[i] * y[i];

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, d->cart_comm);
    return global;
}

double vec_norm2(int n, const double *x, const Decomp *d)
{
    return sqrt(vec_dot(n, x, x, d));
}

// halo_exchange
//
// Shared helper used by matvec_poisson2d, restrict2d, and prolongate2d.
// Packs the four boundary strips of x into send buffers, does a non-blocking
// exchange with all four neighbours, and waits for completion.
// The caller owns the recv buffers and uses them to apply boundary stencils.
// Sends and receives to MPI_PROC_NULL are silently ignored by MPI, so physical
// boundary ranks are handled correctly without any special-casing here.
void halo_exchange(const double *x, int nx, int ny, double *recv_left, double *recv_right, double *recv_bottom, double *recv_top, const Decomp *d)
{
    memset(recv_left, 0, ny * sizeof(double));
    memset(recv_right, 0, ny * sizeof(double));
    memset(recv_bottom, 0, nx * sizeof(double));
    memset(recv_top, 0, nx * sizeof(double));
    
    //Allocate send buffers.
    double *send_left = malloc(ny * sizeof(double));
    double *send_right = malloc(ny * sizeof(double));
    double *send_bottom = malloc(nx * sizeof(double));
    double *send_top = malloc(nx * sizeof(double));

    //Pack send buffers.
    for (int i = 0; i < ny; i++) {
        send_left[i] = x[i * nx + 0]; // leftmost column
        send_right[i] = x[i * nx + (nx - 1)]; // rightmost column
    }
    for (int j = 0; j < nx; j++) {
        send_bottom[j] = x[0 * nx + j]; // bottom row
        send_top[j] = x[(ny - 1) * nx + j]; // top row
    }

    //Non-blocking sends and receives in all four directions.
    MPI_Request reqs[8];
    MPI_Isend(send_left, ny, MPI_DOUBLE, d->neighbours[0], 0, d->cart_comm, &reqs[0]);
    MPI_Irecv(recv_right, ny, MPI_DOUBLE, d->neighbours[1], 0, d->cart_comm, &reqs[1]);
    MPI_Isend(send_right, ny, MPI_DOUBLE, d->neighbours[1], 1, d->cart_comm, &reqs[2]);
    MPI_Irecv(recv_left, ny, MPI_DOUBLE, d->neighbours[0], 1, d->cart_comm, &reqs[3]);
    MPI_Isend(send_bottom, nx, MPI_DOUBLE, d->neighbours[2], 2, d->cart_comm, &reqs[4]);
    MPI_Irecv(recv_top, nx, MPI_DOUBLE, d->neighbours[3], 2, d->cart_comm, &reqs[5]);
    MPI_Isend(send_top, nx, MPI_DOUBLE, d->neighbours[3], 3, d->cart_comm, &reqs[6]);
    MPI_Irecv(recv_bottom, nx, MPI_DOUBLE, d->neighbours[2], 3, d->cart_comm, &reqs[7]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

    //Free send buffers — recv buffers are owned by the caller.
    free(send_left);  
    free(send_right);
    free(send_bottom); 
    free(send_top);
}

// matvec_poisson2d
//
// Applies the 2D Poisson matrix A to the local vector x and writes the
// result into y. Because interior points on the boundary of the local
// subdomain need values from neighbouring ranks, we first do a halo exchange
// then apply the 5-point stencil locally.
//
// Local flat index (no halo): k = i * nx + j
//   i = 0..ny-1 (y-direction, row)
//   j = 0..nx-1 (x-direction, column)
// where nx = d->n_local[1], ny = d->n_local[0].
void matvec_poisson2d(const double *x, double *y, const Decomp *d)
{
    int nx = d->n_local[1]; // local points in x-direction (columns)
    int ny = d->n_local[0]; // local points in y-direction (rows)
    double h2inv = 1.0 / (d->h * d->h);

    //Allocate recv buffers and do the halo exchange.
    double *recv_left = malloc(ny * sizeof(double));
    double *recv_right = malloc(ny * sizeof(double));
    double *recv_bottom = malloc(nx * sizeof(double));
    double *recv_top = malloc(nx * sizeof(double));
    halo_exchange(x, nx, ny, recv_left, recv_right, recv_bottom, recv_top, d);

    //Apply the 5-point stencil. For points on the boundary of the local subdomain
    //use the received halo values. For points on the physical boundary
    //(neighbour == MPI_PROC_NULL) the halo value is zero (homogeneous Dirichlet
    //for the matvec; actual BCs live in the RHS).
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int k = i * nx + j;
            double val = 4.0 * x[k];

            // left neighbour
            if (j > 0)
                val -= x[i * nx + (j - 1)];
            else if (d->neighbours[0] != MPI_PROC_NULL)
                val -= recv_left[i];
            // right neighbour
            if (j < nx - 1)
                val -= x[i * nx + (j + 1)];
            else if (d->neighbours[1] != MPI_PROC_NULL)
                val -= recv_right[i];
            // bottom neighbour
            if (i > 0)
                val -= x[(i - 1) * nx + j];
            else if (d->neighbours[2] != MPI_PROC_NULL)
                val -= recv_bottom[j];
            // top neighbour
            if (i < ny - 1)
                val -= x[(i + 1) * nx + j];
            else if (d->neighbours[3] != MPI_PROC_NULL)
                val -= recv_top[j];

            y[k] = h2inv * val;
        }
    }

    //Free the dynamically allocated halo buffers.
    free(recv_left);  
    free(recv_right);
    free(recv_bottom); 
    free(recv_top);
}
