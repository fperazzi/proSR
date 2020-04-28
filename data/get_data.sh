#! /bin/bash

DATADIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $DATADIR/datasets $DATADIR/checkpoints
sudo apt -y install unzip

##################################
# Pretrained Models
##################################

# ProSRs
ProSRs=$(ls $DATADIR'/checkpoints' | grep proSRs.pth)
if [ $ProSRs == '' ]; then
    wget -nc https://www.dropbox.com/s/ldv397lcr3vn95w/proSRs.zip -O /tmp/proSRs.zip
    unzip -j /tmp/proSRs.zip -d $DATADIR/checkpoints
    rm /tmp/proSRs.zip
fi

# ProSR
ProSR=$(ls $DATADIR'/checkpoints' | grep proSR.pth)
if [ $ProSR == '' ]; then
    wget -nc https://www.dropbox.com/s/3fjp5dd70wuuixl/proSR.zip -O /tmp/proSR.zip
    unzip -j /tmp/proSR.zip -d $DATADIR/checkpoints
    rm /tmp/proSR.zip
fi

# ProSRGAN
ProSRGAN=$(ls $DATADIR'/checkpoints' | grep proSRGAN.pth)
if [ $ProSRGAN == '' ]; then
    wget -nc https://www.dropbox.com/s/ulkvm4yt5v3vxd8/proSRGAN.zip -O /tmp/proSRGAN.zip
    unzip -j /tmp/proSRGAN.zip -d $DATADIR/checkpoints
    rm /tmp/proSRGAN.zip
fi

###################################
## Datasets
###################################
mkdir -p $DATADIR/datasets/DIV2K

## DIV2K train HR
DIV2K_train_HR=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_train_HR)
if [ $DIV2K_train_HR == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P /tmp
    unzip /tmp/DIV2K_train_HR.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_train_HR.zip
fi

## DIV2K valid HR
DIV2K_valid_HR=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_valid_HR)
if [ $DIV2K_valid_HR == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P /tmp
    unzip /tmp/DIV2K_valid_HR.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_valid_HR.zip
fi

## DIV2K train 8x
DIV2K_train_LR_bicubic=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_train_LR_bicubic)
if [ $DIV2K_train_LR_bicubic == '' ]; then
    wget -nc https://www.dropbox.com/s/uvwtxy5hul90hyl/DIV2KX8.zip -O /tmp/DIV2KX8.zip
    unzip /tmp/DIV2KX8.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2KX8.zip
fi

## DIV2K train LR bicubic x4
DIV2K_train_LR_bicubic=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_train_LR_bicubic)
if [ $DIV2K_train_LR_bicubic == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip -P /tmp
    unzip /tmp/DIV2K_train_LR_bicubic_X2.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_train_LR_bicubic_X4.zip
fi

## DIV2K valid LR bicubic x4
DIV2K_valid_LR_bicubic=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_valid_LR_bicubic)
if [ $DIV2K_valid_LR_bicubic == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip -P /tmp
    unzip /tmp/DIV2K_valid_LR_bicubic_X2.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_valid_LR_bicubic_X4.zip
fi

## DIV2K train LR bicubic x2
DIV2K_train_LR_bicubic=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_train_LR_bicubic)
if [ $DIV2K_train_LR_bicubic == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip -P /tmp
    unzip /tmp/DIV2K_train_LR_bicubic_X4.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_train_LR_bicubic_X2.zip
fi

## DIV2K train LR bicubic x2
DIV2K_valid_LR_bicubic=$(ls $DATADIR'/datasets/DIV2k' | grep DIV2K_valid_LR_bicubic)
if [ $DIV2K_valid_LR_bicubic == '' ]; then
    wget -nc http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip -P /tmp
    unzip /tmp/DIV2K_valid_LR_bicubic_X4.zip -d $DATADIR/datasets/DIV2K
    rm /tmp/DIV2K_valid_LR_bicubic_X2.zip
fi

# Flickr2K
Flickr2K_LR_bicubic=$(ls $DATADIR'/datasets/Flickr2K' | grep Flickr2K_LR_bicubic)
if [ $Flickr2K_LR_bicubic == '' ]; then
    wget -nc http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar -P /tmp
    tar -xvf /tmp/Flickr2K.tar -C $DATADIR/datasets/ --keep-newer-files
    rm -rf /tmp/Flickr2K.tar
fi

# Set14, Urban100, BSD100
Urban100=$(ls $DATADIR'/datasets' | grep Urban100)
Set14=$(ls $DATADIR'/datasets' | grep Set14)
BSD100=$(ls $DATADIR'/datasets' | grep BSD100)
if [ $Urban100 == '' ] && [ $Set14 == '' ] && [ $BSD100 == '' ]; then
    wget -nc https://cv.snu.ac.kr/research/EDSR/benchmark.tar -P /tmp
    tar -xvf /tmp/benchmark.tar -C /tmp --keep-newer-files
    mv /tmp/benchmark/* $DATADIR/datasets
    rm -rf /tmp/benchmark
fi