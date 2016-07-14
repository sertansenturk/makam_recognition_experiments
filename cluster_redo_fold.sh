#!/bin/sh
# Name the process
# ----------------
#$ -N makam_$JOB_ID
#
# Call from the current working directory; no need to cd
# ------------------------------------------------------
#$ -cwd
# -q default.q
#
# Max time limits
# ---------------
#$ -l s_rt=5:00:00
#$ -l h_rt=10:00:00
#
# Output/Error Text
# ----------------
#$ -o ../../makam.$JOB_ID.out
#$ -e ../../makam.$JOB_ID.err
#
# Create an array job = !!!!!!number of audio in the target folder!!!!!!
# ----------------
#$ -t 1-19900:1
#
# Send me a mail when processed and when finished:
# ------------------------------------------------
#$ -M sertan.senturk@upf.edu
#$ -m bea

# Start script
# --------------------------------
#

# force UTF 8
export LANG="en_US.utf8"

module load python/2.7.5
module load essentia/2.1_python-2.7.5

python ./redo_fold.py ${SGE_TASK_ID}
