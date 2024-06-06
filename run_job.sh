#!/bin/bash
#PBS -W group_list=jinliu -A jinliu
#PBS -N kernel_indep_test
#PBS -e kernel_indep_test.err
#PBS -o kernel_indep_test.log
#PBS -m n
#PBS -l nodes=1:thinnode:ppn=20
#PBS -l mem=32gb
#PBS -l walltime=01:00:00

echo "Starting job on `date`"
echo "Running on node `hostname`"
echo "Job ID is $PBS_JOBID"

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load necessary modules
module load tools
module load anaconda3/2023.09-0

# Execute the Python script
python run_script.py
