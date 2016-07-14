#!/bin/sh
#$ -N sequentialLinkingDemo
#$ -cwd
#$ -o makam.$JOB_ID.out
#$ -e makam.$JOB_ID.err
# Send me a mail when processed and when finished:
# ------------------------------------------------
#$ -m bea
#$ -M  sertan.senturk@upf.edu
#

# Start script
# --------------------------------
#
printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME\n"
printf "Starting at `date`\n"
printf "---------------------------\n"

# force UTF 8
export LANG="en_US.utf8"
echo $LANG
ls

ipython ./redo_fold.py ${SGE_TASK_ID}

# Copy data back, if any
printf "---------------------------\n"
printf "Job done. Ending at `date`\n"
