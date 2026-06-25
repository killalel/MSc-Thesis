#!/bin/bash
#SBATCH -J mpi_dipole
#SBATCH -o dipole_gridconv_8ranks.out
#SBATCH -e dipole_gridconv_8ranks.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR" || exit 1
module load mpi/2021.12

make clean && make PROBLEM=dipole
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

for N in 63 127 255 511 1023 2047 4095; do
    echo "--- 8 ranks, n=$N, dipole, multigrid ---"
    mpirun -np 8 ./pcg_solver $N 1e-8 10000 multigrid
done