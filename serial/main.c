#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  

#include "poisson2d.h"
#include "linalg.h"
#include "precond.h"
#include "pcg.h"
#include "multigrid.h"

//This is our main file for the serial implementation of the PCG solver for the 2D Poisson problem. It parses command line arguments for 
//the grid size, tolerance, maximum iterations, and preconditioner choice, then sets up the problem, calls the PCG solver, and reports the
// results. The main function is structured to be clear and modular, with separate sections for argument parsing, problem setup, solving,
// and reporting.

//This function just prints the usage message for the program, which is called if the user provides invalid arguments.
static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <n_int> [tol] [max_iter] [precond: blockjacobi|multigrid|multigrid_gs|multigrid_rb|multigrid_cheb|multigrid_sor|multigrid_pci|none]\n", prog);
}

//Here is our main
int main(int argc, char **argv)
{
    //This is just the argument parsing section, which reads in the grid size (n_int), tolerance, maximum iterations, and 
    //preconditioner choice from the command line arguments. It also does some basic error checking on the arguments and prints a 
    //usage message if they are invalid.
    if (argc < 2) { print_usage(argv[0]); return 1; }

    //This is the number of interior grid points in each direction.
    int n_int = atoi(argv[1]);
    //This is the desired tolerance for convergence of the PCG solver. It defaults to 1e-8 if not provided.
    double tol = (argc >= 3) ? atof(argv[2]) : 1e-8;
    //This is the maximum number of iterations for the PCG solver. It defaults to 10*n_int^2 if not provided.
    int max_iter = (argc >= 4) ? atoi(argv[3]) : 10 * n_int * n_int;
    //This is the choice of preconditioner, which can be "blockjacobi", "multigrid", "multigrid_gs", "multigrid_rb", "multigrid_cheb", "multigrid_sor", or "none" (for unpreconditioned CG). It defaults to 
    //"blockjacobi" if not provided.
    char *precond_name = (argc >= 5) ? argv[4] : "blockjacobi";

    //This just prints an error message if n_int < 1.
    if (n_int < 1) {
        fprintf(stderr, "Error: n_int must be >= 1\n");
        return 1;
    }

    //The total number of interior grid points, which is the size of our linear system.
    int N = n_int * n_int;
    //Here we print some of the details of the problem setup.
    printf("=== Serial PCG Poisson Solver ===\n");
    printf("Grid:        %d x %d interior points  (N = %d)\n", n_int, n_int, N);
    printf("h:           %.6e\n", 1.0/(n_int+1));
    printf("Tolerance:   %.2e\n", tol);
    printf("Max iter:    %d\n", max_iter);
    printf("Precond:     %s\n\n", precond_name);

    //Here we allocate memory for and build the rhs.
    double *b = malloc(N * sizeof(double));
    build_rhs(n_int, b);

    //Here we allocate memory for the solution vector and set the initial guess to zero.
    double *x = calloc(N, sizeof(double));

    //Here we set up the preconditioner based on the user's choice. We use function pointers to allow for different preconditioners to 
    //be used with the same PCG solver. See precond.c/.h for more details on the preconditioner implementations.
    PrecondApplyFn  precond_apply;
    void *precond_ctx;
    BlockJacobiCtx *bj_ctx  = NULL;
    IdentityCtx *id_ctx  = NULL;
    MultigridCtx *mg_ctx = NULL;

    //This is for timing the program.
    clock_t setup_start = clock(); 

    //Here we check the preconditioner choice and set up the appropriate context and function pointer for applying the preconditioner.
    if (strcmp(precond_name, "blockjacobi") == 0) {
        bj_ctx = block_jacobi_setup(n_int);
        precond_apply = block_jacobi_apply;
        precond_ctx = bj_ctx;
    } else if (strcmp(precond_name, "multigrid") == 0) {
        mg_ctx = multigrid_setup(n_int, 2, 2, 2.0/3.0, wj_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "multigrid_gs") == 0) {   
        mg_ctx = multigrid_setup(n_int, 2, 2, 1.0, gs_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    }  else if (strcmp(precond_name, "multigrid_rb") == 0) {
        mg_ctx = multigrid_setup(n_int, 2, 2, 1.0, rb_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "multigrid_cheb") == 0) {
        mg_ctx = multigrid_setup(n_int, 2, 2, 1.0, cheb_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "multigrid_sor") == 0) {
        mg_ctx = multigrid_setup(n_int, 2, 2, 1.3, sor_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "multigrid_pci") == 0) {
        mg_ctx = multigrid_setup(n_int, 2, 2, 1.0, pci_smooth);
        precond_apply = multigrid_apply;
        precond_ctx = mg_ctx;
    } else if (strcmp(precond_name, "none") == 0) {
        id_ctx = identity_setup(n_int);
        precond_apply = identity_apply;
        precond_ctx = id_ctx;
    } else {
        fprintf(stderr, "Unknown preconditioner: %s\n", precond_name);
        print_usage(argv[0]);
        return 1;
    }

    //This ends theb timing of the setup
    clock_t setup_end = clock(); 

    //Here we allocate memory for the array to store the relative residual history, which will be printed after the solve.
    double *rel_res = malloc((max_iter + 1) * sizeof(double));

    //This starts the timing of the solve.
    clock_t solve_start = clock();

    //Here we call the PCG solver, which returns the number of iterations taken to converge. The 
    int iters = pcg_solve(n_int, b, x, tol, max_iter, precond_apply, precond_ctx, rel_res);

    //This ends the timing of the solve.
    clock_t solve_end = clock();

    //This is for reporting the results after the solve. We print whether PCG converged and the final relative residual.
    if (iters < 0) {
        printf("WARNING: PCG did not converge in %d iterations.\n", max_iter);
        printf("Final relative residual: %.6e\n\n", rel_res[max_iter]);
    } else {
        printf("PCG converged in %d iterations.\n", iters);
        printf("Final relative residual: %.6e\n\n", rel_res[iters]);
    }

    //This gets the maximum pointwise error of the computed solution x against the analytic solution at the interior grid points.
    double err = max_error(n_int, x);
    printf("Max pointwise error vs analytic solution: %.6e\n\n", err);

    //This computes and prints the timing results for the setup and solve phases of the program.
    double setup_time = (double)(setup_end - setup_start) / CLOCKS_PER_SEC;
    double solve_time = (double)(solve_end - solve_start) / CLOCKS_PER_SEC;
    printf("Setup time:  %.6f s\n", setup_time);
    printf("Solve time:  %.6f s\n", solve_time);
    printf("Total time:  %.6f s\n\n", setup_time + solve_time);


    //This prints the convergence history.
    int n_print = (iters < 0) ? max_iter : iters;
    printf("Iteration | Rel. Residual\n");
    printf("----------+--------------\n");
    for (int k = 0; k <= n_print; k++)
        printf("  %6d  |  %.6e\n", k, rel_res[k]);

    //This frees the dynamically allocated memory.
    if (bj_ctx) block_jacobi_free(bj_ctx);
    if (mg_ctx) multigrid_free(mg_ctx); 
    if (id_ctx) identity_free(id_ctx);
    free(rel_res);
    free(b);
    free(x);

    return (iters < 0) ? 1 : 0;
}