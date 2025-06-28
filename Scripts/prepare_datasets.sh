#!/bin/bash
# The 3RScan dataset should be downloaded and extracted to the ../Datasets/3RScan directory
set -euxo pipefail


# get 3RScan metadata
mkdir ../Datasets/3RScan/3DSSG
mv ../Datasets/3RScan/data/3RScan.json ../Datasets/3RScan
mv ../Datasets/3RScan/data/objects.json ../Datasets/3RScan/3DSSG
rm ../Datasets/3RScan/data/relationships.json

cd ../Datasets/3RScan
wget http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip
unzip 3DSSG_subset.zip
rm 3DSSG_subset.zip
cd ../../scripts

# check integrity
python dataset/check.py --path ../Datasets/3RScan

# copy the 3DSSG to ScanNet categories mapping, which can be found here: https://docs.google.com/spreadsheets/d/1eRTJ2M9OHz7ypXfYD-KTR1AIT-CrVLmhJf8mxgVZWnI/edit?usp=sharing
cp files/3dssg_to_scannet.json ../Datasets/3RScan/3DSSG_subset/3dssg_to_scannet.json

# copy our train/val split since the original mixed up reference scans and rescans
cp files/train_scans.txt ../Datasets/3RScan/3DSSG_subset/train_scans.txt
cp files/validation_scans.txt ../Datasets/3RScan/3DSSG_subset/validation_scans.txt

# We use the validation split in 3RScan as the test split
wget https://campar.in.tum.de/public_datasets/3RScan/val_scans.txt -O ../Datasets/3RScan/3DSSG_subset/test_scans.txt

# extract scannet classes annotations from 3RScan dataset
python dataset/relationship2scannet.py --path ../Datasets/3RScan

# extract 2D bounding boxes from 3RScan dataset
python dataset/boxes2coco.py --path ../Datasets/3RScan --label_categories scannet

# sanity checks
python tools/show_bbox.py --path ../Datasets/3RScan
python tools/show_sg_labels.py --path ../Datasets/3RScan