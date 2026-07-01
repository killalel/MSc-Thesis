#!/bin/bash
#SBATCH -J mpi_sm_dip_full
#SBATCH -o sm_full_dipole.out
#SBATCH -e sm_full_dipole.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR" || exit 1
module load mpi/2021.12

make clean && make PROBLEM=dipole
if [ $? -ne 0 ]; then
    echo "Compilation failed - stopping job."
    exit 1
fi

for sm in multigrid multigrid_gs multigrid_rb multigrid_sor; do
    for ranks in 1 2 4 8 16; do
        for N in 63 127 255 511 1023 2047 4095 8191 16383 32767; do
            if [ $N -ge 16383 ] && [ $ranks -lt 4 ]; then
                echo "--- skipping $ranks ranks, n=$N, dipole, $sm (memory/time) ---"
                continue
            fi
            if [ $N -ge 32767 ] && [ $ranks -lt 4 ]; then
                continue
            fi
            echo "--- $ranks ranks, n=$N, dipole, $sm ---"
            mpirun -np $ranks ./pcg_solver $N 1e-8 10000 $sm
        done
    done
done
