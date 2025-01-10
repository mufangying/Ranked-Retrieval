#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=CRC_varying_beta    # Assign an short name to your job
#SBATCH --array=0-0
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                  # Real memory (RAM) required (MB)
#SBATCH --time=72:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=output_file/tcrc_varying_beta.%N.%j.out     # STDOUT output file
#SBATCH --error=output_file/tcrc_varying_beta.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env


srun python3 vary_beta_mslr.py