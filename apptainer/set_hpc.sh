#!/bin/bash

# Configuration
if [ -z "$walltime" ]; then
    walltime=24:00:00
    echo WARNING: Default value walltime=24:00:00
fi
if [ -z "$ncpus" ]; then
    ncpus=8
    echo WARNING: Default value ncpus=8
fi
if [ -z "$mem" ]; then
    mem=24gb
    echo WARNING: Default value mem=24gb
fi
if [ -z "$ngpus" ]; then
    ngpus=1
    echo WARNING: Default value ngpus=1
fi


# Define variables
project_name=${PWD##*/}
image_path=$1
image_name=${image_path##*/}
image_directory=${image_name%.*}
job_name=$2
commit=$(echo $image_directory | cut -d "_" -f 4)

# Check number of arguments
if [ $# -lt 2 ]; then
    echo ERROR: Invalid number of arguments.
    exit 1
fi

# Check if the path is valid
if [ ! -f $image_path ]; then
    echo ERROR: Invalid Apptainer image file.
    exit 1
fi

# Manage GPU
if [ $ngpus -gt 0 ]; then
    gpu=":ngpus=$ngpus:gpu_type=RTX6000"
fi

# Create temporary directory
TMP_DIR=$(mktemp -d -p apptainer/)

# Save git log output to track container
git log --decorate --color -10 $commit > $TMP_DIR/git-log.txt

# Build job script from template
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
sed "s/@job_name/$job_name/g" $SCRIPT_DIR/template.job > $TMP_DIR/script.job
sed -i "s/@project_name/$project_name/g" $TMP_DIR/script.job
sed -i "s/@image_directory/$image_directory/g" $TMP_DIR/script.job
sed -i "s/@image_name/$image_name/g" $TMP_DIR/script.job
sed -i "s/@app/$job_name/g" $TMP_DIR/script.job
sed -i "s/@commit/$commit/g" $TMP_DIR/script.job
sed -i "s/@args/${@:3}/g" $TMP_DIR/script.job
sed -i "s/@walltime/$walltime/g" $TMP_DIR/script.job
sed -i "s/@ncpus/$ncpus/g" $TMP_DIR/script.job
sed -i "s/@mem/$mem/g" $TMP_DIR/script.job
sed -i "s/@gpu/$gpu/g" $TMP_DIR/script.job
sed -i "s/@sync/$sync/g" $TMP_DIR/script.job
mv $TMP_DIR/script.job $TMP_DIR/$job_name.job

# Send Apptainer image and job script to HPC
ssh hpc "mkdir -p $project_name/; mkdir $project_name/$image_directory/;"
if [ $? == 1 ]; then
    echo WARNING: A Singularity image file with the same name is already on the HPC. Copying only the job script.
    rsync --verbose --progress -e ssh $TMP_DIR/$job_name.job hpc:$project_name/$image_directory/
else
    rsync --verbose --progress -e ssh $image_path $TMP_DIR/git-log.txt $TMP_DIR/$job_name.job hpc:$project_name/$image_directory/
fi

rm -rf $TMP_DIR
