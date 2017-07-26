#!/bin/sh
#
#$ -S /bin/sh
#$ -N GPflow_Cocoex_Test_01
#$ -wd /srv/global/scratch/aryanmash
#$ -j y
#$ -l h_slots=1
#$ -l s_rt=10:00:00
#$ -l virtual_free=8G

#if [ -d /srv/local/work/$JOB_ID ]; then
#        cd /srv/local/work/$JOB_ID
if [ -d /srv/local/work/$JOB_ID.$SGE_TASK_ID ]; then
        cd /srv/local/work/$JOB_ID.$SGE_TASK_ID
else
        echo "There's no job directory to change into "
        echo "Here's LOCAL TMP "
        ls -la /srv/local/work
        echo "Exiting"
        exit 1
fi
#
# Now we are in the job-specific directory so
#
module load scl-python/27
module load java/1.8.0
module load scl-gcc/4.9.2
#
gpflowcoco=/srv/global/scratch/groups/secs/GPflowCoco
PYTHONPATH=$gpflowcoco/lib/python2.7/site-packages
export PYTHONPATH
export PATH=$gpflowcoco/bin:$PATH
#
#  Copy the input data over
cp -a $GLOBAL_SCRATCH/ExpOpt .
# 
cd ExpOpt
python experiments_dropoutneto.py $1  $2 $SGE_TASK_ID $3  $4 $5
#
# Now we move the output to a place to pick it up from later
#  (really should check that directory exists too, but this is just a test)
#
mkdir -p $GLOBAL_SCRATCH/exdata
cp -pr exdata/* $GLOBAL_SCRATCH/exdata/
#
