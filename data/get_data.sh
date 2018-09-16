#! /bin/bash

DATADIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $DATADIR/datasets $DATADIR/checkpoints

##################################
# Pretrained Models
##################################

# ProSRs
wget https://www.dropbox.com/s/ldv397lcr3vn95w/proSRs.zip?dl=0 -O /tmp/proSRs.zip
unzip -j /tmp/proSRs.zip -d $DATADIR/checkpoints && rm /tmp/proSRs.zip

# ProSR
wget https://www.dropbox.com/s/3fjp5dd70wuuixl/proSR.zip?dl=0 -O /tmp/proSR.zip
unzip -j /tmp/proSR.zip -d $DATADIR/checkpoints && rm /tmp/proSR.zip

# ProSRGAN
wget https://www.dropbox.com/s/ulkvm4yt5v3vxd8/proSRGAN.zip?dl=0 -O /tmp/proSRGAN.zip
unzip -j /tmp/proSRGAN.zip -d $DATADIR/checkpoints && rm /tmp/proSRGAN.zip

###################################
## Datasets
###################################

## DIV2K
mkdir -p $DATADIR/datasets/DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P /tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P /tmp
unzip /tmp/DIV2K_train_HR.zip -d $DATADIR/datasets/DIV2K
unzip /tmp/DIV2K_valid_HR.zip -d $DATADIR/datasets/DIV2K
rm /tmp/DIV2K_train_HR.zip
rm /tmp/DIV2K_valid_HR.zip

wget https://www.dropbox.com/s/uvwtxy5hul90hyl/DIV2KX8.zip?dl=0 -O /tmp/DIV2KX8.zip
unzip /tmp/DIV2KX8.zip -d $DATADIR/datasets/DIV2K
rm /tmp/DIV2KX8.zip

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip -P /tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip -P /tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip -P /tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip -P /tmp

unzip /tmp/DIV2K_train_LR_bicubic_X2.zip -d $DATADIR/datasets/DIV2K
unzip /tmp/DIV2K_valid_LR_bicubic_X2.zip -d $DATADIR/datasets/DIV2K
unzip /tmp/DIV2K_train_LR_bicubic_X4.zip -d $DATADIR/datasets/DIV2K
unzip /tmp/DIV2K_valid_LR_bicubic_X4.zip -d $DATADIR/datasets/DIV2K

rm /tmp/DIV2K_train_LR_bicubic_X4.zip
rm /tmp/DIV2K_valid_LR_bicubic_X4.zip
rm /tmp/DIV2K_train_LR_bicubic_X2.zip
rm /tmp/DIV2K_valid_LR_bicubic_X2.zip

# Flickr2K
# wget http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar -P /tmp
# tar -xvf /tmp/Flickr2K.tar -C $DATADIR/datasets/
# rm -rf /tmp/Flickr2K.tar

# Set14, Urban100, BSD100
wget https://cv.snu.ac.kr/research/EDSR/benchmark.tar -P /tmp
tar -xvf /tmp/benchmark.tar -C /tmp && mv /tmp/benchmark/* $DATADIR/datasets
rm -rf /tmp/benchmark /tmp/benchmark.tar
