# ProSR

Unofficial implementation of [A Fully Progressive Approach to Single-Image Super-Resolution](https://fperazzi.github.io/files/publications/prosr.pdf).

![](data/prosr-teaser.jpg)

ProSR is a Single Image Super-Resolution (SISR) method designed upon the principle of multi-scale progressiveness. The architecture resembles an asymmetric pyramidal structure with more layers in the upper levels to enable high upsampling ratios while remaining efficient. The training procedure implements the paradigm of curriculum learning by gradually increasing the difficulty of the task.
![](data/prosr-arch.jpg)

## Gettings Started
Follow the intrunctions below to get ProSR up and running on your machine, for developement and testing purposes.

### System Requirements
*ProSR* is developed under Ubuntu 16.04 with CUDA 9.1, cuDNN v7.0 and pytorch-0.4.0. We tested the program on Nvidia Titan X and Tesla K40c GPUs. Any NVIDIA GPU with ~12GB memory will do. Parallel processing on multiple GPUs will be supported during training.

### Dependencies
  * python 3.x
  * pytorch 0.4.0
  * cuda91
  * torch
  * torchvision
  * scikit-image
  * pillow
  * easydict

#### Install Dependencies
```
# Crate virtual environment
conda create -n proSR

# Install torch
conda install pytorch=0.4.0 torchvision cuda91 -c pytorch

# Install image libraries
conda install scikit-image pillow 

# Install pip and easydict
conda install pip && pip install easydict

# Additional modules for visualization
conda install visdom dominate -c conda-forge
```

#### Search Path

`export PYTHONPATH=$PROJECT_ROOT/lib:$PYTHONPATH` to include `proSR` into the search path. 

## Data
In `PROJECT_ROOT/data` we provide a script `get_data.sh` to download the a pretrained model for x8 upsampling. 
TLDR; Download the data: `sh data/get_data.sh`

### Pretrained Models
We provide the following pretrained models:

* [ProSR](https://www.dropbox.com/s/hlgunvtmkvylc4h/proSR.pth?dl=0) - This the full size model that ranked 4th place in terms of PSNR and second when measured with SSIM on the "Track 1" of the [NTIRE Super-Resolution Challenge 2018](https://competitions.codalab.org/competitions/18015).
* [ProSRs]() - A lightweight version of ProSR. Best speed / accuracy tradeoff.
* [ProSRGAN]() - ProSR trained with an adversarial loss. Lower PSNR but higher details.

### Datasets

We pretrained ProSR on DIV2K and Flickr2K. We evaluated the results on the following datasets:


## Testing
Run `test.py`

```
usage: test.py [-h] -c CHECKPOINT -i INPUT [INPUT ...]
               [-t TARGET [TARGET ...]] -u UPSCALE_FACTOR [-f FMT]
               [-o OUTPUT_DIR]
```
optional arguments:
```
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Checkpoint
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input images, either list or path to folder
  -t TARGET [TARGET ...], --target TARGET [TARGET ...]
                        Target images, either list or path to folder
  -u UPSCALE_FACTOR, --upscale-factor UPSCALE_FACTOR
                        List of images to upsample
  -f FMT, --fmt FMT     Image file format
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output folder.
```


By default, the output images will be saved `/tmp/<class_name>` where `<class_name>` is the name of the architecture defined in the `checkpoints['params'][class_name]`.



### Quickstart
Excute the following commands to upsample images provided in `$PROJECT_ROOT/data/examples`
```
# Upsample image by 8 times and save result in '/tmp'
python test.py --checkpoint data/checkpoints/proSR.pth -i data/examples/0801x8.png

# Upsample images in folder by factor of 4, evaluate
# results (SSIM and PSNR) and save results in /tmp/prosr

python test.py --checkpoint data/checkpoints/proSR.pth \
  -i data/examples -t data/examples/0801 -u 4 -o /tmp/prosr_examples
```

### Reproduce results
To reproduce the results reported in table 1, first you need to download the data as explained in section X.


# Results 
| Model  | S14 | B100 | U100 | DIV2K | S14 | B100 | U100 | DIV2K | S14| B100 | U100 | DIV2K |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
MsLapSRN | 33.28 | 32.05 | 31.15 | 35.62 | 28.26 | 27.43 | 25.51 | 30.39 | 24.57 | 24.65 | 22.06 | 26.52 |
| EDSR   | 33.92 | 32.32 | 32.93 | 36.47 | 28.80 | 27.71 | 26.64 | 30.71 | 24.96 | 24.83 | 22.53 | 26.96 |
[ProSRs]() | 33.36 | 32.02 | 31.42 | 35.80 | 28.59 | 27.58 | 26.01 | 30.39 | 24.93 | 24.80 | 22.43 | 26.88 |
[ProSR](https://www.dropbox.com/s/hlgunvtmkvylc4h/proSR.pth?dl=0)   | 34.00 | 32.34 | 32.91 | 36.44 | 28.94 | 27.79 | 26.89 | 30.81 | 25.29 | 24.99 | 23.04 | 27.36 |



## Training
Not implemented yet. Send an email to [fperazzi@adobe.com](fpearzzi@adobe.com) if you want to be notified when available.

### Configuration
The available options are defined in `lib/prosr/config.py`.

## Publication
If this code helps your research, please considering citing the following paper.

A Fully Progressive Approach to Single-Image Super-Resolution - <i>[Y. Wang](https://yifita.github.io), [F. Perazzi](fperazzi.github.io), [B. McWilliams](https://www.inf.ethz.ch/personal/mcbrian/), [A. Sorkine-Hornung](http://www.ahornung.net/), [O. Sorkine-Hornung](http://igl.ethz.ch/people/sorkine/), [C. Schroers](https://www.disneyresearch.com/people/christopher-schroers/)</i> - CVPR Workshops NTIRE 2018.
```
@InProceedings{Wang_2018_CVPR_Workshops,
    author = {
      Wang, Y. and
      Perazzi, F. and
      McWilliams, B. and
      Sorkine-Hornung, A. and
      Sorkine-Hornung, O and
      Schroers, C.},
  title = {A Fully Progressive Approach to Single-Image Super-Resolution},
  booktitle = {CVPR Workshops},
  month = {June},
  year = {2018}
}
```

### Contacts
If you have any question, please contact [Yifan Wang](yifan.wang@inf.ethz.ch) and [Federico Perazzi](fperazzi@adobe.com).

### License
Figure out something

## Todolist
 [ ] Finish README
 [ ] Remove scikit-image dependency?
 [ ] Restructure parameters and merge in test/train from commandline
 [ ] Add training
 [ ] Refactoring printing/logging


