#include <stdlib.h>
#include <stdio.h>
#include "decomp.h"

Decomp *decomp_create(int n_global)
{
    Decomp *d = malloc(sizeof(Decomp));
    d->n_global = n_global;
    d->h = 1.0 / (n_global + 1);

    MPI_Comm_size(MPI_COMM_WORLD, &d->size);

    // Let MPI choose a balanced 2D process grid factorisation.
    // dims[0] is the number of process rows (y-direction),
    // dims[1] is the number of process columns (x-direction).
    d->dims[0] = 0;
    d->dims[1] = 0;
    MPI_Dims_create(d->size, 2, d->dims);

    // Create a 2D Cartesian communicator with no periodicity.
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, d->dims, periods, 1, &d->cart_comm);
    MPI_Comm_rank(d->cart_comm, &d->rank);
    MPI_Cart_coords(d->cart_comm, d->rank, 2, d->coords);

    // Distribute n_global points as evenly as possible across each dimension.
    // Ranks with a lower coordinate index get one extra point if n_global
    // doesn't divide evenly.
    for (int dim = 0; dim < 2; dim++) {
        int base  = n_global / d->dims[dim];
        int extra = n_global % d->dims[dim];
        // coords[dim] < extra ranks get base+1 points, the rest get base.
        d->n_local[dim]  = base + (d->coords[dim] < extra ? 1 : 0);
        // Global offset: sum of n_local for all ranks before this one.
        d->offsets[dim]  = d->coords[dim] * base + (d->coords[dim] < extra ? d->coords[dim] : extra);
    }

    // Find the four neighbour ranks using MPI_Cart_shift.
    // MPI_PROC_NULL is returned for ranks on the physical boundary.
    MPI_Cart_shift(d->cart_comm, 1, 1, &d->neighbours[0], &d->neighbours[1]); // left/right (x)
    MPI_Cart_shift(d->cart_comm, 0, 1, &d->neighbours[2], &d->neighbours[3]); // bottom/top (y)

    return d;
}

//Free the Decomp structure and its associated MPI communicator.
void decomp_free(Decomp *d)
{
    MPI_Comm_free(&d->cart_comm);
    free(d);
}

//Print the decomposition information for rank 0.
void decomp_print(const Decomp *d)
{
    if (d->rank != 0) return;
    printf("=== Decomposition ===\n");
    printf("Global grid:   %d x %d interior points\n", d->n_global, d->n_global);
    printf("Process grid:  %d x %d\n", d->dims[0], d->dims[1]);
    printf("h:             %.6e\n", d->h);
    printf("MPI ranks:     %d\n\n", d->size);
}