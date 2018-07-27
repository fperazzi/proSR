#! /bin/bash

DATADIR="$( cd ../"$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $DATADIR/{datasets,checkpoints}

##################################
# Pretrained Models
##################################
wget https://www.dropbox.com/s/hj1rcew430l6cbf/EDSR.pth?dl=0 -P $DATADIR/checkpoints
wget https://www.dropbox.com/s/hlgunvtmkvylc4h/proSR.pth?dl=0 -P $DATADIR/checkpoints

# DIV2K
# wget https://cv.snu.ac.kr/research/EDSR/DIV2K.tar -P /tmp/
# mkdir -p $DATADIR/datasets/DIV2K
# tar -xvf /tmp/DIV2K.tar -C $DATADIR/datasets/DIV2K
# rm -rf /tmp/DIV2K.tar

# Flickr2K
# wget http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar -P /tmp
# tar -xvf /tmp/Flickr2K.tar -C $DATADIR/datasets/
# rm -rf /tmp/Flickr2K.tar

