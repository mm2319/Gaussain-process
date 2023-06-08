#!/bin/sh

# request bash as shell for job
#PBS -S /bin/bash

# queue, parallel environment and number of processors
#PBS -l select=1:ncpus=5:mem=10gb

#PBS -l walltime=72:00:00

# joins error and standard outputs
#PBS -j oe

# keep error and standard outputs on the execution host
#PBS -k oe

# forward environment variables
#PBS -V

# define job name
#PBS -N GP

# main commands here

module load anaconda3/personal

cd /rds/general/user/mm2319/home/Gaussain-process/

python3 GP_lr_3_train_noise_001.py