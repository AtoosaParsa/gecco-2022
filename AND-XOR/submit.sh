#!/bin/bash
# Specify partition
#SBATCH --partition=bluemoon
# Request nodes
#SBATCH --nodes=1
# Request processor cores
#SBATCH --ntasks=51
#SBATCH --cpus-per-task=1
# Request memory for the entire job
# Remember, you can request --mem OR --mem-per-cpu, but not both
#SBATCH --mem=10G
# Reserve walltime
#SBATCH --time=24:00:00
# Name this job
#SBATCH --job-name=MOO
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

# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Prepare in the current folder a worker launcher for Scoop 
# The scipt below will 'decorate' the python interpreter command
# Before python is called, modules are loaded
HOSTFILE=$(pwd)/hostfile
SCOOP_WRAPPER=$(pwd)/scoop-python.sh

cat << EOF > $SCOOP_WRAPPER
#!/bin/bash -l
export SLURM_NTASKS=${SLURM_NTASKS}
EOF
echo 'python $@' >> $SCOOP_WRAPPER

chmod +x $SCOOP_WRAPPER

# Classical "module load" in the main script

# Save the hostname of the allocated nodes
scontrol show hostnames > $HOSTFILE

# Start scoop with python input script
INPUTFILE=$(pwd)/MOO.py 
python -m scoop --hostfile $HOSTFILE -n ${SLURM_NTASKS} --python-interpreter=$SCOOP_WRAPPER $INPUTFILE $@