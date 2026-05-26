#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "precond.h"




//This is just a simple identity preconditioner, which does nothing. It just copies the input vector r to the output vector z.
//This is for testing purposes.
IdentityCtx *identity_setup(int n_int)
{
    IdentityCtx *ctx = malloc(sizeof(IdentityCtx));
    ctx->n = n_int * n_int;
    return ctx;
}
 
// M = I, so solving M*z = r is just z = r. We use memcpy to copy the vector efficiently.
void identity_apply(double *z, const double *r, void *ctx)
{
    IdentityCtx *ic = (IdentityCtx *)ctx;
    memcpy(z, r, ic->n * sizeof(double));
}
 
void identity_free(IdentityCtx *ctx)
{
    free(ctx);
}

//This is the block Jacobi preconditioner for the 2D Poisson problem. The preconditioner is block diagonal, where each block corresponds 
//to a grid row (x-direction) and is the tridiagonal matrix T from the 1D Poisson problem. We use the Thomas algorithm to solve T z = r 
//for each block, which requires some setup to compute the modified diagonal and multipliers for forward elimination. The setup function 
//allocates a context structure to store these factors, and the apply function uses them to perform the forward and back substitution for 
//each block.
BlockJacobiCtx *block_jacobi_setup(int n_int)
{
    //This is just the context structure for the block Jacobi preconditioner.
    BlockJacobiCtx *ctx = malloc(sizeof(BlockJacobiCtx));
    ctx->n_int = n_int;
    ctx->h = 1.0 / (n_int + 1);

    //This is just 1/h^2
    double h2inv = 1.0 / (ctx->h * ctx->h);

    //Here we allocate one set of Thomas factors (length n_int each) 
    ctx->d = malloc(n_int * sizeof(double));
    ctx->l = malloc(n_int * sizeof(double));   

    //This just initialises the diagonal 
    for (int i = 0; i < n_int; i++)
        ctx->d[i] = 2.0 * h2inv;

    ctx->l[0] = 0.0;

    //This is the Thomas forward elimination on the single block T
    double sub = -h2inv; //This is the sub- and super-diagonal value 
    for (int i = 1; i < n_int; i++) {
        ctx->l[i] = sub / ctx->d[i-1]; //This is the multiplier for the forward elimination step.
        ctx->d[i] -= ctx->l[i] * sub; //This is the modified diagonal after forward elimination.
    }

    //We return the context, which contains the factors needed for applying the preconditioner.
    return ctx;
}

//Here we apply the block Jacobi preconditioner. We loop over each block (grid row), perform forward substitution to solve L w = r for 
//the current block,
void block_jacobi_apply(double *z, const double *r, void *ctx)
{
    //
    BlockJacobiCtx *bj = (BlockJacobiCtx *)ctx;
    int n = bj->n_int;
    double h2inv = 1.0 / (bj->h * bj->h);
    double sub  = -h2inv;

    //Here we loop over each block (grid row) and apply the Thomas algorithm to solve T z = r for that block.
    for (int blk = 0; blk < n; blk++) {
        //Pointer to the start of this block in r and z 
        const double *rb = r + blk * n;
        double *zb = z + blk * n;

        //Forward substitution: L * w = rb, storing result in zb 
        zb[0] = rb[0];
        for (int i = 1; i < n; i++)
            zb[i] = rb[i] - bj->l[i] * zb[i-1];

        //Back substitution: U * z = w 
        zb[n-1] /= bj->d[n-1];
        for (int i = n-2; i >= 0; i--)
            zb[i] = (zb[i] - sub * zb[i+1]) / bj->d[i];
    }
}

//Here we free all of the dynamically allocated memory.
void block_jacobi_free(BlockJacobiCtx *ctx)
{
    free(ctx->d);
    free(ctx->l);
    free(ctx);
}