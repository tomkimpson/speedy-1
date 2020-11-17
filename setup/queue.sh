#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set the number of processes per node.
#SBATCH --ntasks-per-node=1

# set max wallclock time (d-h:m:s)
#SBATCH --time=24:00

# set partition name
#SBATCH --partition=priority-predict

# set name of job
#SBATCH --job-name=speedy

# mail alert at start, end, and abortion of execution
###SBATCH --mail-type=ALL

# send mail to this address
###SBATCH --mail-user=my.address.here@physics.ox.ac.uk

# run the application
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/network/aopp/chaos/pred/shared/madam/rpe_complex/lib/

time ./imp.exe | tee out.lis

touch DONE