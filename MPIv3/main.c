#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#include "decomp.h"
#include "poisson2d.h"
#include "linalg.h"
#include "precond.h"
#include "pcg.h"

//This function prints the usage message for the program, indicating the required and optional command-line arguments.
static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <n_global> [tol] [max_iter] [precond: blockjacobi|multigrid|none]\n", prog);
}

int main(int argc, char **argv)
{
    //Initialize MPI.
    MPI_Init(&argc, &argv);

    //Get the rank of the current process in MPI_COMM_WORLD.
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //If the number of command-line arguments is less than 2, print the usage message and exit.
    if (argc < 2) {
        if (world_rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    //Parse command-line arguments.
    int n_global = atoi(argv[1]);
    double tol = (argc >= 3) ? atof(argv[2]) : 1e-8;
    int max_iter = (argc >= 4) ? atoi(argv[3]) : 10 * n_global * n_global;
    char *precond_name = (argc >= 5) ? argv[4] : "blockjacobi";

    //If n_global is less than 1, print an error message and exit.
    if (n_global < 1) {
        if (world_rank == 0) fprintf(stderr, "Error: n_global must be >= 1\n");
        MPI_Finalize();
        return 1;
    }

    //Set up the 2D Cartesian decomposition.
    Decomp *d = decomp_create(n_global);

    //Set the number of local unknowns on this rank.
    int N = d->n_local[0] * d->n_local[1];

    //The 0th rank prints the problem setup information.
    if (world_rank == 0) {
        printf("=== MPI PCG Poisson Solver ===\n");
        printf("Global grid:  %d x %d interior points\n", n_global, n_global);
        printf("Process grid: %d x %d\n", d->dims[0], d->dims[1]);
        printf("MPI ranks:    %d\n", d->size);
        printf("h:            %.6e\n", d->h);
        printf("Tolerance:    %.2e\n", tol);
        printf("Max iter:     %d\n", max_iter);
        printf("Precond:      %s\n\n", precond_name);
    }

    // Build local RHS and zero initial guess.
    double *b = malloc(N * sizeof(double));
    double *x = calloc(N, sizeof(double));
    build_rhs(b, d);

    // Set up preconditioner.
    PrecondApplyFn  precond_apply;
    void *precond_ctx;
    BlockJacobiCtx *bj_ctx = NULL;
    IdentityCtx *id_ctx = NULL;
    MultigridCtx *mg_ctx = NULL;

    double setup_start = MPI_Wtime();

    // Choose preconditioner based on command-line argument.
    if (strcmp(precond_name, "blockjacobi") == 0) {
        bj_ctx = block_jacobi_setup(d);
        precond_apply = block_jacobi_apply;
        precond_ctx = bj_ctx;
    } else if (strcmp(precond_name, "multigrid") == 0) {
        mg_ctx = multigrid_setup(d, 2, 2, 2.0/3.0, wj_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "none") == 0) {
        id_ctx = identity_setup(N);
        precond_apply = identity_apply;
        precond_ctx = id_ctx;
    } else {
        if (world_rank == 0) {
            fprintf(stderr, "Unknown preconditioner: %s\n", precond_name);
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Measure setup time.
    double setup_end = MPI_Wtime();

    // Solve.
    double *rel_res = malloc((max_iter + 1) * sizeof(double));

    // Measure solve time.
    double solve_start = MPI_Wtime();
    int iters = pcg_solve(b, x, tol, max_iter, precond_apply, precond_ctx, rel_res, d);
    double solve_end = MPI_Wtime();

    //This computes the maximum pointwise error between the computed solution x and the analytic solution.
    double err = max_error(x, d);

    // Report from rank 0.
    if (world_rank == 0) {
        if (iters < 0) {
            printf("WARNING: PCG did not converge in %d iterations.\n", max_iter);
            printf("Final relative residual: %.6e\n\n", rel_res[max_iter]);
        } else {
            printf("PCG converged in %d iterations.\n", iters);
            printf("Final relative residual: %.6e\n\n", rel_res[iters]);
        }

        printf("Max pointwise error vs analytic solution: %.6e\n\n", err);

        printf("Setup time:  %.6f s\n", setup_end - setup_start);
        printf("Solve time:  %.6f s\n", solve_end - solve_start);
        printf("Total time:  %.6f s\n\n", (setup_end - setup_start) + (solve_end - solve_start));

        int n_print = (iters < 0) ? max_iter : iters;
        printf("Iteration | Rel. Residual\n");
        printf("----------+--------------\n");
        for (int k = 0; k <= n_print; k++)
            printf("  %6d  |  %.6e\n", k, rel_res[k]);
    }

    // Cleanup.
    if (bj_ctx) block_jacobi_free(bj_ctx);
    if (mg_ctx) multigrid_free(mg_ctx);
    if (id_ctx) identity_free(id_ctx);
    free(rel_res);
    free(b);
    free(x);
    decomp_free(d);

    MPI_Finalize();
    return (iters < 0) ? 1 : 0;
}
