#!/bin/bash
# Specify partition
#SBATCH --partition=bluemoon
# Request nodes
#SBATCH --nodes=1
# Request processor cores
#SBATCH --ntasks=128
# Request memory for the entire job
# Remember, you can request --mem OR --mem-per-cpu, but not both
#SBATCH --mem=10G
# Reserve walltime
#SBATCH --time=24:00:00
# Name this job
#SBATCH --job-name=XORRSv3DiffK
# Name output of this job using %x=job-name and %j=job-id
#SBATCH --output=%x_%j.out 

#SBATCH --mail-user=aparsa1@uvm.edu
#SBATCH --mail-type=ALL

#
# Change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
#
# For fun, echo some useful and interesting information 
echo "Starting sbatch script myscript.sh at:`date`"
echo "  running host:    ${SLURMD_NODENAME}"
echo "  assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "  partition used:  ${SLURM_JOB_PARTITION}"
echo "  jobid:           ${SLURM_JOBID}"
#
# As an example, load spack python3 and run the script hello.py
python randomSearch.py

