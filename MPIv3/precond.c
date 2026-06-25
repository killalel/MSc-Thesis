#include <stdlib.h>
#include <string.h>
#include "precond.h"

// Identity preconditioner
IdentityCtx *identity_setup(int n)
{
    IdentityCtx *ctx = malloc(sizeof(IdentityCtx));
    ctx->n = n;
    return ctx;
}

// M = I, so solving M*z = r is just z = r. We use memcpy to copy the vector efficiently.
void identity_apply(double *z, const double *r, void *ctx)
{
    IdentityCtx *ic = (IdentityCtx *)ctx;
    memcpy(z, r, ic->n * sizeof(double));
}

void identity_free(IdentityCtx *ctx) { free(ctx); }

// Block Jacobi preconditioner.
// nx = d->n_local[1] is the block size (x-direction points per block).
// ny = d->n_local[0] is the number of blocks (one per local grid row).
// Each rank factors one set of Thomas factors (all blocks are identical)
// and applies them to all ny blocks independently — no communication needed.
BlockJacobiCtx *block_jacobi_setup(const Decomp *d)
{
    //Here we allocate the context structure for the block Jacobi preconditioner and set the parameters based on the decomposition.
    BlockJacobiCtx *ctx = malloc(sizeof(BlockJacobiCtx));
    ctx->nx = d->n_local[1];
    ctx->ny = d->n_local[0];
    ctx->h  = d->h;

    int n = ctx->nx;
    //1/h^2
    double h2inv = 1.0 / (d->h * d->h);
    double sub  = -h2inv;

    //allocate memory for the Thomas factors for one block (length n each)
    ctx->d = malloc(n * sizeof(double));
    ctx->l = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
        ctx->d[i] = 2.0 * h2inv;
    ctx->l[0] = 0.0;

    for (int i = 1; i < n; i++) {
        ctx->l[i]  = sub / ctx->d[i-1];
        ctx->d[i] -= ctx->l[i] * sub;
    }

    return ctx;
}

//This function applies the block Jacobi preconditioner to the input vector r, storing the result in z. 
void block_jacobi_apply(double *z, const double *r, void *ctx)
{
    BlockJacobiCtx *bj   = (BlockJacobiCtx *)ctx;
    int nx = bj->nx;
    int ny = bj->ny;
    double h2inv = 1.0 / (bj->h * bj->h);
    double sub = -h2inv;

    for (int blk = 0; blk < ny; blk++) {
        const double *rb = r + blk * nx;
        double *zb = z + blk * nx;

        zb[0] = rb[0];
        for (int i = 1; i < nx; i++)
            zb[i] = rb[i] - bj->l[i] * zb[i-1];

        zb[nx-1] /= bj->d[nx-1];
        for (int i = nx-2; i >= 0; i--)
            zb[i] = (zb[i] - sub * zb[i+1]) / bj->d[i];
    }
}

//Free memory
void block_jacobi_free(BlockJacobiCtx *ctx)
{
    free(ctx->d);
    free(ctx->l);
    free(ctx);
}

// Multigrid preconditioner.
// multigrid_setup builds the full grid hierarchy from the fine Decomp down to
// the coarsest level where n_global == 1. Work arrays are pre-allocated for
// each level so the V-cycle doesn't malloc/free on every call.
MultigridCtx *multigrid_setup(const Decomp *d, int nu1, int nu2,
                              double omega, SmootherFn smoother)
{
    MultigridCtx *ctx = malloc(sizeof(MultigridCtx));
    ctx->nu1 = nu1;
    ctx->nu2 = nu2;
    ctx->omega = omega;
    ctx->smoother = smoother;

    //Count levels by coarsening until n_global == 1.
    int n_levels = 1;
    int n = d->n_global;
    while (n > 1) {
        n = (n - 1) / 2;
        n_levels++;
    }
    ctx->n_levels = n_levels;

    ctx->decomps = malloc(n_levels * sizeof(Decomp *));
    ctx->work_u = malloc(n_levels * sizeof(double *));
    ctx->work_f = malloc(n_levels * sizeof(double *));
    ctx->work_r = malloc(n_levels * sizeof(double *));

    //Level 0 is the fine grid — we don't own this Decomp, just reference it.
    ctx->decomps[0] = (Decomp *)d;
    int N0 = d->n_local[0] * d->n_local[1];
    ctx->work_u[0] = malloc(N0 * sizeof(double));
    ctx->work_f[0] = malloc(N0 * sizeof(double));
    ctx->work_r[0] = malloc(N0 * sizeof(double));

    //Build coarser levels — each level gets its own Decomp and work arrays.
    n = d->n_global;
    for (int lev = 1; lev < n_levels; lev++) {
        n = (n - 1) / 2;
        ctx->decomps[lev] = decomp_create(n);
        int N = ctx->decomps[lev]->n_local[0] * ctx->decomps[lev]->n_local[1];
        ctx->work_u[lev] = malloc(N * sizeof(double));
        ctx->work_f[lev] = malloc(N * sizeof(double));
        ctx->work_r[lev] = malloc(N * sizeof(double));
    }

    return ctx;
}

// multigrid_apply
//
// Applies one V-cycle as the preconditioner: approximately solves M*z = r.
// z is zeroed before calling vcycle so the V-cycle acts as a linear operator,
// which is required for correctness inside PCG.
void multigrid_apply(double *z, const double *r, void *ctx)
{
    MultigridCtx *mg = (MultigridCtx *)ctx;
    int N = mg->decomps[0]->n_local[0] * mg->decomps[0]->n_local[1];

    //Zero initial guess — essential for linearity.
    memset(z, 0, N * sizeof(double));

    vcycle(0, mg->n_levels, mg->decomps, mg->work_u, mg->work_f, mg->work_r, z, r, mg->omega, mg->nu1, mg->nu2, mg->smoother);
}

//Free all memory owned by the multigrid context.
//Level 0 Decomp is owned by the caller so we don't free it.
void multigrid_free(MultigridCtx *ctx)
{
    for (int lev = 0; lev < ctx->n_levels; lev++) {
        free(ctx->work_u[lev]);
        free(ctx->work_f[lev]);
        free(ctx->work_r[lev]);
        if (lev > 0) decomp_free(ctx->decomps[lev]);
    }
    free(ctx->decomps);
    free(ctx->work_u);
    free(ctx->work_f);
    free(ctx->work_r);
    free(ctx);
}
