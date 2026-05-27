#ifndef PRECOND_H
#define PRECOND_H

/*
 * precond.h
 *
 * This is the header file for our preconditioner implementations. It defines the interface for preconditioners used in the PCG solver, 
 * //as well as the specific context structures and function prototypes for the block Jacobi and identity preconditioners.
 * 
 * New preconditioners can be added by defining a new context structure, setup function, apply function, and free function, 
 * following the pattern of the existing preconditioners.
 * 
 * More detailed comments can be found in the corresponding precond.c file, which contains the implementations of these functions.
 */

//This is a generic function pointer type for applying a preconditioner. The actual preconditioner is determined by the context struct 
//that is passed in, which is cast to the appropriate type inside the apply function.

/*
 * PrecondApplyFn: solves M*z = r.
 *   z   [out] solution vector (length n)
 *   r   [in]  right-hand side vector (length n)
 *   ctx [in]  preconditioner context (cast inside the function)
 */
typedef void (*PrecondApplyFn)(double *z, const double *r, void *ctx);


/* ------------------------------------------------------------------ */
/* This section defines the Block Jacobi preconditioner                                         */
/* ------------------------------------------------------------------ */


//This is the structure for the block Jacobi preconditioner context. 
typedef struct {
    int    n_int;    //interior grid points per direction          
    double h;        //mesh spacing 1/(n_int+1)                    
    double *l;       //lower diag of LU factors, length n_int*n_int (Thomas) 
    double *d;       // modified diagonal after elimination          
} BlockJacobiCtx;


BlockJacobiCtx *block_jacobi_setup(int n_int);


void block_jacobi_apply(double *z, const double *r, void *ctx);


void block_jacobi_free(BlockJacobiCtx *ctx);


/* ------------------------------------------------------------------ */
/* Identity preconditioner (unpreconditioned CG, useful for testing)  */
/* ------------------------------------------------------------------ */

//This is the structure for the identity preconditioner context. 
typedef struct {
    int n;    //length of the vectors (n_int*n_int)
} IdentityCtx;

IdentityCtx *identity_setup(int n_int);
void identity_apply(double *z, const double *r, void *ctx);
void identity_free(IdentityCtx *ctx);

/* ------------------------------------------------------------------ */
/* Multigrid preconditioner                                            */
/* ------------------------------------------------------------------ */

#include "multigrid.h"

//This is the structure for the multigrid preconditioner context. 
typedef struct {
    int    n_int;   //Number of interior grid points per direction on the fine grid.
    double h;       //fine grid mesh spacing
    int    nu1;     //No. of pre-smoothing steps
    int    nu2;     //No. of post-smoothing steps
    double omega;   //Jacobi damping weight
} MultigridCtx;

//Function prototypes for the multigrid preconditioner setup, apply, and free functions.
MultigridCtx *multigrid_setup(int n_int, int nu1, int nu2, double omega);
void          multigrid_apply(double *z, const double *r, void *ctx);
void          multigrid_free(MultigridCtx *ctx);

#endif /* PRECOND_H */