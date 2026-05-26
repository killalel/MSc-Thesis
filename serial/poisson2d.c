#include <math.h>
#include <stdlib.h>
#include "poisson2d.h"

/* 
*
* poisson2d.c
*
*This is our main file for the 2D Poisson problem. It contains functions for building the right-hand side vector from the boundary 
*conditions, computing the maximum error against the analytic solution, and the analytic solution itself. The main function is in a
*separate file, main.c, which calls these functions to set up and solve the problem, and then compute and print the error.
*/


//These are the boundary value functions. They take the coordinate of the boundary point and return the value of the boundary 
//condition at that point. They are declared as static since they are only used in this file. 
//Note that they were chosen to mirror those used in HPC Software HW1.
static double bc_left(double y) { return y / (1.0 + y*y); }
static double bc_right(double y) { return y / (4.0 + y*y); }
static double bc_bottom(double x) { (void)x; return 0.0; }
static double bc_top(double x) { return 1.0 / ((1.0+x)*(1.0+x) + 1.0); }

//This function computes the analytic solution at a given interior grid point (i,j).
double analytic_solution(int i, int j, int n_int)
{
    double h = 1.0 / (n_int + 1);
    double x = i * h;
    double y = j * h;
    return y / ((1.0 + x)*(1.0 + x) + y*y);
}

//This is our function for building the right hand side vector b.
void build_rhs(int n_int, double *b)
{
    //This i sjust the grid spacing.
    double h = 1.0 / (n_int + 1);
    //This i sjust 1/h^2
    double h2inv = 1.0 / (h * h);

    //Here we set all internal points to zero 
    for (int k = 0; k < n_int * n_int; k++)
        b[k] = 0.0;

    //Here we loop over all interior grid points, compute the contributions from the boundary conditions, and write to b.
    for (int i = 1; i <= n_int; i++) {
        for (int j = 1; j <= n_int; j++) {

            int k = (i-1)*n_int + (j-1);   // 0-based flat index 
            double x = i * h; //This is just the x coordinate of the grid point, which we need for the boundary conditions.
            double y = j * h; //This is just the y coordinate of the grid point, which we need for the boundary conditions.

            // left boundary neighbour at (i-1, j) = (0, j) 
            if (i == 1)
                b[k] += h2inv * bc_left(y);

            // right boundary neighbour at (i+1, j) = (n_int+1, j) 
            if (i == n_int)
                b[k] += h2inv * bc_right(y);

            // bottom boundary neighbour at (i, j-1) = (i, 0) 
            if (j == 1)
                b[k] += h2inv * bc_bottom(x);

            // top boundary neighbour at (i, j+1) = (i, n_int+1) 
            if (j == n_int)
                b[k] += h2inv * bc_top(x);
        }
    }
}

//This function computes the maximum error of the computed solution x against the analytic solution at the interior grid points.
double max_error(int n_int, const double *x)
{
    double err = 0.0;
    for (int i = 1; i <= n_int; i++) {
        for (int j = 1; j <= n_int; j++) {
            int k = (i-1)*n_int + (j-1); // 0-based flat index
            double diff = fabs(x[k] - analytic_solution(i, j, n_int)); //The difference between the computed solution and the analytic solution at this grid point.
            if (diff > err) err = diff; //Update the maximum error if this point has a larger error than the current maximum.
        }
    }
    //Returns the error
    return err;
}