#ifndef MULTIGRID_H
#define MULTIGRID_H

//This is the header file for our multigrid implementation. More detailed comments can be found in the corresponding multigrid.c file.


//This is the function pointer type for smoothers.
typedef void (*SmootherFn)(int n_int, double h, double omega, int nu, double *u, const double *f);

void wj_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);
 
void gs_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);

void rb_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);

void cheb_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);

void sor_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);

void restrict2d(int n_fine, const double *fine, double *coarse);
 

void prolongate2d(int n_coarse, const double *coarse, int n_fine, double *fine);
 

void vcycle(int n_int, double h, double omega, int nu1, int nu2, SmootherFn smoother, double *u, const double *f);
 
#endif 
 