#!/bin/bash
#SBATCH --job-name=b0_correction
#SBATCH --output=b0_correction_%A_%a.out
#SBATCH --error=b0_correction_%A_%a.err
#SBATCH --account=simurgh
#SBATCH --partition=visionlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120000M
#SBATCH --time=24:00:00

export FSLOUTPUTTYPE=NIFTI_GZ

# Define paths
#SUBJECT_LIST="/simurgh/u/gustavochau/dti_processing/list_subjects_adni"
SUBJECT_LIST="/simurgh/group/BWM/DTISherlock/hcpag/imagingcollection01/subject_list.txt"

# Count number of subjects
NUM_SUBJECTS=$(wc -l < $SUBJECT_LIST)

# Read subject ID from the list
SUBJECT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" $SUBJECT_LIST)

echo "Processing subject: $SUBJECT_ID"

python -u b0_correction.py $SUBJECT_ID

echo "Done B0 correction!"
