#!/bin/bash
#SBATCH -J mpi_smoothers
#SBATCH -o smoothers_1023_8ranks.out
#SBATCH -e smoothers_1023_8ranks.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR" || exit 1
module load mpi/2021.12

make clean && make PROBLEM=original
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

for sm in multigrid multigrid_gs multigrid_rb multigrid_sor; do
    echo "--- 8 ranks, n=1023, $sm ---"
    mpirun -np 8 ./pcg_solver 1023 1e-8 10000 $sm
done
