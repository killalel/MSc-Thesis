#ifndef DECOMP_H
#define DECOMP_H

#include <mpi.h>

// This file holds all the information about how the global grid is distributed
// across MPI processes. Every function that needs to know about the parallel
// layout takes a const Decomp* rather than having MPI calls scattered everywhere.
//
// The global interior grid is n_global x n_global points. It is distributed
// across a dims[0] x dims[1] process grid. Each rank owns a contiguous
// n_local[0] x n_local[1] block of interior points.
//
// Flat index convention within the local patch (including one layer of halo):
// k = (i + 1) * (n_local[1] + 2) + (j + 1)
// where i runs 0..n_local[0]-1 and j runs 0..n_local[1]-1.
// The +1 offsets skip the halo layer. Halo points sit at i=-1, i=n_local[0],
// j=-1, j=n_local[1] in logical coordinates.

typedef struct {
    MPI_Comm cart_comm; // 2D Cartesian communicator
    int rank; // this rank's index in cart_comm
    int size; // total number of MPI processes
    int coords[2]; // this rank's (row, col) in the process grid
    int dims[2]; // process grid dimensions: dims[0] x dims[1]
    int n_local[2]; // local interior points in each direction
    int n_global; // global interior points per direction
    double h; // mesh spacing 1/(n_global + 1)
    int offsets[2]; // global index of first local interior point
    int neighbours[4]; // neighbour ranks: 0=left 1=right 2=bottom 3=top
                        // MPI_PROC_NULL if on physical boundary
} Decomp;

// Allocates and initialises a Decomp for an n_global x n_global interior grid
// distributed across all processes in MPI_COMM_WORLD. MPI_Dims_create is used
// to find a balanced dims[0] x dims[1] factorisation of the process count.
// Returns a heap-allocated Decomp; caller frees with decomp_free().
Decomp *decomp_create(int n_global);

// Frees the Decomp and its communicator.
void decomp_free(Decomp *d);

// Prints a summary of the decomposition from rank 0.
void decomp_print(const Decomp *d);

#endif // DECOMP_H