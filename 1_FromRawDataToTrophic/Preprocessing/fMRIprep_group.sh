#!/bin/bash
#SBATCH --job-name=ADNI_Preprocessing
#SBATCH -p high
#SBATCH -N 1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-$(($(ls $HOME/ADNI_dataset | grep sub | wc -l)-1))

TASK='rest'
DATASET_DIR="${HOME}/ADNI3_dataset"
OUTPUT_DIR="${HOME}/ADNI3_preprocessed"
WORKDIR="${OUTPUT_DIR}/workdir"
LICENSE_FILE="${HOME}/license.txt"
FMRIPREP_IMAGE='/soft/singularity/fmriprep-21.0.1.simg'
THREADS=8
MEM_GB=32

task='rest'

ID=${SLURM_ARRAY_TASK_ID}
mapfile -t SUBS < <(find "$DATASET_DIR" -maxdepth 1 -type d -name "sub-*" | sort)
readonly PARTICIPANT="${SUBS[$ID]##*/}"

echo "Processing: $PARTICIPANT (Task $ID of ${#SUBS[@]})"

singularity run --cleanenv \
 -B $DATASET_DIR:/data \
 -B $OUTPUT_DIR:/output \
 -B $LICENSE_FILE:/license.txt \
 $FMRIPREP_IMAGE \
 /data /output participant \
 --participant-label $PARTICIPANT \
 -t $TASK \
 --work-dir $WORKDIR \
 --nthreads $THREADS \
 --omp-nthreads $THREADS \
 --mem $MEM_GB \
 --fs-license-file $LICENSE_FILE \
 --skip-bids-validation

echo "Completed processing: $PARTICIPANT"


