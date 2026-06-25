#!/bin/bash
#SBATCH -J mpi_scaling
#SBATCH -o scaling_original_1023.out
#SBATCH -e scaling_original_1023.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR" || exit 1
module load mpi/2021.12

make clean && make PROBLEM=original
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

N=1023
for ranks in 1 2 4 8 16; do
    echo "--- $ranks ranks, n=$N, multigrid ---"
    mpirun -np $ranks ./pcg_solver $N 1e-8 10000 multigrid
    echo "--- $ranks ranks, n=$N, blockjacobi ---"
    mpirun -np $ranks ./pcg_solver $N 1e-8 10000 blockjacobi
done