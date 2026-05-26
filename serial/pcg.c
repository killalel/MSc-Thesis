#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pcg.h"
#include "linalg.h"

/*
 * pcg.c
 *
 * This file is the implementation of the Preconditioned Conjugate Gradient solver.
 * This is based on Saad Algorithm 9.1 (2nd edition) and Case Studies HW3, extended with preconditioning.
 */

int pcg_solve(int n_int, const double *b,double *x, double tol, int max_iter, PrecondApplyFn precond_apply, void *precond_ctx, double *rel_res)
{
    //This is just the total number of interior grid points, which is the size of our linear system.
    int N = n_int * n_int;

    //Here we allocate worskpace vectors.
    double *r = malloc(N * sizeof(double)); //This is the residual vector, r = b - A x.
    double *z = malloc(N * sizeof(double)); //This is the preconditioned residual, z = M^{-1} r, where M is our preconditioner.
    double *p = malloc(N * sizeof(double)); //This is the search direction vector, which is updated each iteration.
    double *Ap = malloc(N * sizeof(double)); //This is the temporary vector for storing A*p each iteration.            

    //This is the grid spacing
    double h = 1.0 / (n_int + 1);

    //First we compute the initial residual r0 = b - A x0, where x0 is our initial guess (stored in x).
    matvec_poisson2d(n_int, h, x, r);       
    //Now r contains A x, so we do r = b - r to get the residual.
    for (int i = 0; i < N; i++)
        r[i] = b[i] - r[i];                   

    //Now we apply the preconditioner to the initial residual to get z0 = M^{-1} r0.
    precond_apply(z, r, precond_ctx);

    //Here we set the initial search direction p0 = z0.
    vec_copy(N, z, p);

    //This is just the norm of the right-hand side.
    double b_norm = vec_norm2(N, b);

    //Here we store the initial relative residual norm.
    rel_res[0] = vec_norm2(N, r) / b_norm;

    //This is the initial value of (r,z) for the first iteration.
    double rz = vec_dot(N, r, z);

    //This is just a variable to count iterations.
    int iter;

    for (iter = 0; iter < max_iter; iter++) {

        //Here we use our function for computing the matrix-vector product Ap = A p, from linalg.c
        matvec_poisson2d(n_int, h, p, Ap);

        //Here we compute the step size alpha = (r,z) / (p,Ap). The numerator is just the current value of (r,z) stored in rz
        // and the denominator is computed using the dot product function from linalg.c.
        double pAp   = vec_dot(N, p, Ap);
        double alpha = rz / pAp;

        // Here we update the solution x = x + alpha * p, using the axpy function from linalg.c.
        vec_axpy(N, alpha, p, x);

        // Here we update the residual r = r - alpha * Ap, using the axpy function from linalg.c.
        vec_axpy(N, -alpha, Ap, r);

        //This just checks the convergence
        double res_norm = vec_norm2(N, r);
        rel_res[iter + 1] = res_norm / b_norm;

        //This is just for the case where the relative residual is below tolerance
        if (rel_res[iter + 1] < tol) {
            iter++;   // count this iteration 
            break;
        }

        // z = M^{-1} r
        precond_apply(z, r, precond_ctx);

        // beta = (r_new, z_new) / (r_old, z_old)
        double rz_new = vec_dot(N, r, z);
        double beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        for (int i = 0; i < N; i++)
            p[i] = z[i] + beta * p[i];
    }

    //Here we free all of the dynamically allocated memory.
    free(r);
    free(z);
    free(p);
    free(Ap);

    //This just Return -1 if we ran out of iterations without converging 
    if (iter == max_iter && rel_res[max_iter] >= tol)
        return -1;

    return iter;
}