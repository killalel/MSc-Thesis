
//This file is for basic linear algebra operations on vectors and matrices.


#include <math.h>
#include "linalg.h"

// This just does y = y + alpha * x 
void vec_axpy(int n, double alpha, const double *x, double *y)
{
    for (int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

//This just copies a vector, y = x
void vec_copy(int n, const double *x, double *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i];
}

//This just sets all entries of a vector to zero, y = 0
void vec_zero(int n, double *x)
{
    for (int i = 0; i < n; i++)
        x[i] = 0.0;
}

//This just computes the dot product of two vectors, s = x^T y
double vec_dot(int n, const double *x, const double *y)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

//This just computes the 2-norm of a vector, ||x||_2 = sqrt(x^T x)
double vec_norm2(int n, const double *x)
{
    return sqrt(vec_dot(n, x, x));
}

//This is our functioon for computing the matrix-vector product y = A x, where A is the 5-point stencil matrix for the 
//2D Poisson problem on a square grid with n_int interior points in each direction and grid spacing h.
void matvec_poisson2d(int n_int, double h, const double *x, double *y)
{
    //This is just 1/h^2
    double h2inv = 1.0 / (h * h);
    //This is just the total number of interior grid points.
    int N = n_int * n_int;

    //Here we loop over all interior grid points, compute the stencil sum, and write to y.
    for (int k = 0; k < N; k++)
    {
        //We compute the 2D grid indices (i,j) from the flat index k.
        int i = k / n_int; //This is just the x direction (column) index.
        int j = k % n_int; //This is just the y direction (row) index.

        //This is the central point contribution, which is 4 * x[k] in the stencil.
        double val = 4.0 * x[k];

        //Left neighbour: (i-1, j)
        if (i > 0)          val -= x[(i-1)*n_int + j];
        //Right neighbour: (i+1, j)
        if (i < n_int - 1)  val -= x[(i+1)*n_int + j];
        //Bottom neighbour: (i, j-1)
        if (j > 0)          val -= x[i*n_int + (j-1)];
        //Top neighbour: (i, j+1)
        if (j < n_int - 1)  val -= x[i*n_int + (j+1)];

        //Finally we scale by 1/h^2 and write to y.
        y[k] = h2inv * val;
    }
}