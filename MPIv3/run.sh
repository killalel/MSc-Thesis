#!/bin/bash
#SBATCH -J mpi_pcg
#SBATCH -o mpi_pcg.out
#SBATCH -e mpi_pcg.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:01:00

cd "$SLURM_SUBMIT_DIR" || exit 1

module load mpi/2021.12

make clean && make
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

echo "--- 4 ranks, n=63, blockjacobi ---"
mpirun -np 4 ./pcg_solver 63 1e-8 10000 blockjacobi

echo "--- 4 ranks, n=63, none ---"
mpirun -np 4 ./pcg_solver 63 1e-8 10000 none
