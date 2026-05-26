#ifndef LINALG_H
#define LINALG_H


//This is the header file for our simple linear algebra library. It contains function prototypes for basic vector operations
// (axpy, copy, zero, dot product, norm) and a matrix-vector product function for the 2D Poisson problem using a 5-point stencil. 
//The actual implementations of these functions are in linalg.c, as well as comments explaining what each function does.


void vec_axpy(int n, double alpha, const double *x, double *y);
 

void vec_copy(int n, const double *x, double *y);
 

void vec_zero(int n, double *x);
 

double vec_dot(int n, const double *x, const double *y);
 
double vec_norm2(int n, const double *x);
 

void matvec_poisson2d(int n_int, double h, const double *x, double *y);
 
#endif 
 