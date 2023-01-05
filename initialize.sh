#!/bin/bash

DATA_ZIP_FILE_NAME=food_c_data.zip
SHARE_DATA_DIR=/share3/amoghtiwari/data/$DATA_ZIP_FILE_NAME
GNODE_DATA_DIR=/ssd_scratch/cvit/amoghtiwari/food_c_challenge/data
GNODE_CKPTS_DIR=/ssd_scratch/cvit/amoghtiwari/food_c_challenge/ckpts

mkdir -p $GNODE_CKPTS_DIR
# rm -r $GNODE_DATA_DIR
mkdir -p $GNODE_DATA_DIR
cd $GNODE_DATA_DIR

echo "Copying Data"
scp amoghtiwari@ada:$SHARE_DATA_DIR ./
echo "Extracting Data"
unzip -q $DATA_ZIP_FILE_NAME
rm $DATA_ZIP_FILE_NAME

cd ~/personal_projects/food_c_challenge
bash utils/create_sample_data.sh
