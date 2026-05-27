#ifndef MULTIGRID_H
#define MULTIGRID_H

//This is the header file for our multigrid implementation. More detailed comments can be found in the corresponding multigrid.c file.
void wj_smooth(int n_int, double h, double omega, int nu, double *u, const double *f);
 

void restrict2d(int n_fine, const double *fine, double *coarse);
 

void prolongate2d(int n_coarse, const double *coarse, int n_fine, double *fine);
 

void vcycle(int n_int, double h, double omega, int nu1, int nu2, double *u, const double *f);
 
#endif 
 