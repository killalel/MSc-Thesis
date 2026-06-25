#!/bin/bash
#SBATCH -J mpi_dipole_scale
#SBATCH -o dipole_scaling_1023.out
#SBATCH -e dipole_scaling_1023.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR" || exit 1
module load mpi/2021.12

make clean && make PROBLEM=dipole
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

for ranks in 1 2 4 8 16; do
    echo "--- $ranks ranks, n=1023, dipole, multigrid ---"
    mpirun -np $ranks ./pcg_solver 1023 1e-8 10000 multigrid
done
