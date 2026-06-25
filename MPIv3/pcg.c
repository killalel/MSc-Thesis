#include <stdlib.h>
#include <math.h>
#include "pcg.h"
#include "linalg.h"

int pcg_solve(const double *b, double *x, double tol, int max_iter, PrecondApplyFn precond_apply, void *precond_ctx, double *rel_res, const Decomp *d)
{
    //Number of local unknowns on this rank.
    int N = d->n_local[0] * d->n_local[1];

    //Allocate temporary vectors needed for the PCG algorithm.
    double *r = malloc(N * sizeof(double));
    double *z = malloc(N * sizeof(double));
    double *p = malloc(N * sizeof(double));
    double *Ap = malloc(N * sizeof(double));

    // r0 = b - A*x0  (x0 = 0 so r0 = b)
    matvec_poisson2d(x, r, d);
    for (int i = 0; i < N; i++)
        r[i] = b[i] - r[i];

    // z0 = M^{-1} r0
    precond_apply(z, r, precond_ctx);

    // p0 = z0
    vec_copy(N, z, p);

    double b_norm = vec_norm2(N, b, d);
    rel_res[0] = vec_norm2(N, r, d) / b_norm;

    //This is the initial value of r^T z, which is used in the first iteration to compute alpha.
    double rz = vec_dot(N, r, z, d);

    int iter;
    for (iter = 0; iter < max_iter; iter++) {

        // Ap = A * p  (includes halo exchange)
        matvec_poisson2d(p, Ap, d);

        double alpha = rz / vec_dot(N, p, Ap, d);

        // x = x + alpha * p
        vec_axpy(N, alpha, p, x);

        // r = r - alpha * Ap
        vec_axpy(N, -alpha, Ap, r);

        double res_norm = vec_norm2(N, r, d);
        rel_res[iter + 1] = res_norm / b_norm;

        if (rel_res[iter + 1] < tol) {
            iter++;
            break;
        }

        // z = M^{-1} r
        precond_apply(z, r, precond_ctx);

        double rz_new = vec_dot(N, r, z, d);
        double beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        for (int i = 0; i < N; i++)
            p[i] = z[i] + beta * p[i];
    }

    //Free dynamically allocated vectors.
    free(r); 
    free(z); 
    free(p); 
    free(Ap);

    if (iter == max_iter && rel_res[max_iter] >= tol)
        return -1;
    return iter;
}