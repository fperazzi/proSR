# ProSR

### [A Fully Progressive Approach to Single-Image Super-Resolution](https://fperazzi.github.io/files/publications/prosr.pdf).
(Unofficial Implementation)

![](docs/figures/prosr-teaser.jpg)

**ProSR** is a Single Image Super-Resolution (SISR) method designed upon the principle of multi-scale progressiveness. The architecture resembles an asymmetric pyramidal structure with more layers in the upper levels to enable high upsampling ratios while remaining efficient. The training procedure implements the paradigm of curriculum learning by gradually increasing the difficulty of the task.

## Installation
Follow the instructions below to get **ProSR** up and running on your machine, both for development and testing purposes.

### System Requirements
**ProSR** is developed under Ubuntu 16.04 with CUDA 9.1, cuDNN v7.0 and pytorch-0.4.0. We tested the program on Nvidia Titan X and Tesla K40c GPUs. Any NVIDIA GPU with ~12GB memory will do. Parallel processing on multiple GPUs is supported during training.

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
conda install scikit-image cython

# Install visdom
conda install visdom dominate -c conda-forge

# Install pip and easydict
pip install easydict html
```

#### Search Path

`export PYTHONPATH=$PROJECT_ROOT/lib:$PYTHONPATH` to include `proSR` into the search path.

## Getting the Data
The script `get_data.sh`, found in `PROJECT_ROOT/data`, downloads the pretrained models and datasets that we used in this project. This is a large download of approximately 10GB that might take a while to complete. If you would rather download individual files, continue reading the next section.

### Datasets
The results reported in the paper are trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) ([7.1GB](https://data.vision.ee.ethz.ch/cvl/DIV2K/)). Improved performance, at the expenses of longer training time can be obtained adding [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) to the training data. The pretrained network available in this repository have been trained with DIVK and Flickr2K.

Furthermore, we evaluated the performance of ProSR on the following benchmark datasets:

* [Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* [Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
* [B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr)

A package containing all the above benchmark datasets was conveniently made available by the [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) colleagues: [benchmark.tar](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

### Pretrained Models
We provide the following pretrained models:

* proSR - This is the full size model that ranked 2nd and 4th place respectively in terms of PSNR and SSIM on the "Track 1" of the [NTIRE Super-Resolution Challenge 2018](https://competitions.codalab.org/competitions/18015).
* proSRs - A lightweight version of ProSR. Best speed / accuracy tradeoff.
* proSRGAN - ProSR trained with an adversarial loss. Lower PSNR but higher details.

The above models performs well across different upscaling ratios [2,4,8]. However, best performance can be achived using scale specific models. These models are available in the same folder and are post-fixed with `_xSCALE` (e.g. `proSR_x8.pth`) to indicate at which regime perform best.

![](docs/figures/prosr-arch.jpg)
## Results
Following wide-spread protocol, the quantitative results are obtained converting RGB images to YCbCr and evaluating the PSNR and SSIM on the Y channel only. Refer to `eval.py` for further details about the evaluation.

| Model  | Params | S14 | B100 | U100 | DIV2K | S14 | B100 | U100 | DIV2K | S14| B100 | U100 | DIV2K |
|--------|-----|------|------|-------|-----|----|----|------|------|-----|----|-----|
|    <td colspan=5> 2x <td colspan=4> 4x <td colspan=4>8x
|MsLapSRN | 0.2M |33.28 | 32.05 | 31.15 | 35.62 | 28.26 | 27.43 | 25.51 | 30.39 | 24.57 | 24.65 | 22.06 | 26.52 |
| EDSR   | 40M | 33.92 | 32.32 | 32.93 | 36.47 | 28.80 | 27.71 | 26.64 | 30.71 | 24.96 | 24.83 | 22.53 | 26.96 |
ProSRs | 3M | 33.36 | 32.02 | 31.42 | 35.80 | 28.59 | 27.58 | 26.01 | 30.39 | 24.93 | 24.80 | 22.43 | 26.88 |
ProSR | 15M | 34.00 | 32.34 | 32.91 | 36.44 | 28.94 | 27.79 | 26.89 | 30.81 | 25.29 | 24.99 | 23.04 | 27.36 |

Colleagues from [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) made available a package containing all of the above datasets: [benchmark.tar](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

## Training
You can train your own model using the script `train.py`:
```
# Train using default params
python train.py --model MODEL --output DIR
```

`MODEL` is one of `prosr` or `prosrs`. Model configurations is loaded from `prosr/config.py`. Checkpoints and log files are stored in `DIR`. Alternatively, we provide configuration files to customize model and training parameters:
```
# Train with configuration file.
python train.py --config CONFIG.yaml
```
Configurations files of the architectures proposed in the paper are avaiable in `PROJECT_ROOT/options`.

### Loading the dataset
Set the path to the dataset in `configs.py:prosr_params.train.path{source,target}`. To **train on multiple datasets** create a new folder containing soft links to the datasets you want to use for training. For example: `ensemble/{DIV2K_train_HR,Flickr2K}`.

`train.path.source` is optional. If left empty, the dataloader will downsample the target images found in `train.path.target` to the predefined lower resolution.

See `train.py` for more options:
```
usage: train.py [-h] (-m {prosr,prosrs,debug} | -c CONFIG | -ckpt CHECKPOINT)
                [--no-curriculum] [-o OUTPUT] [--seed SEED]
                [--fast-validation FAST_VALIDATION] [-v] [-p VISDOM_PORT]
```
optional arguments:
```
  -h, --help            show this help message and exit
  -m {prosr,prosrs,debug}, --model {prosr,prosrs,debug}
                        model
  -c CONFIG, --config CONFIG
                        Configuration file in 'yaml' format.
  -ckpt CHECKPOINT, --checkpoint CHECKPOINT
                        path to previous training session
  --no-curriculum       disable curriculum learning
  -o OUTPUT, --output OUTPUT
                        output folder
  --seed SEED           reproducible experiments
  --fast-validation FAST_VALIDATION
                        truncate number of validation images
  -v, --visdom
  -p VISDOM_PORT, --visdom-port VISDOM_PORT
                        port used by visdom
```
#### Training Paper Experiments
```
# ProSRs (training time 16hrs on 4 NVIDIA Xp)
python train.py --model prosrs --output proSRs

# ProSR (training time 16hrs on 4 NVIDIA Xp)
python train.py --model prosr --output proSR

# ProSRs+ (training time 16hrs on 4 NVIDIA Xp)
python train.py --config options/prosrs+.yaml--output proSR

# ProSR+ (training time 16hrs on 4 NVIDIA Xp)
python train.py --config options/prosr+.yaml--output proSR


```
#### Resume Training
To resume training from a checkpoint, e.g. `data/checkpoints/PRETRAINED_net_G.pth`.
```
python train.py --checkpoint data/checkpoints/PRETRAINED
```

#### MultiGPU Training
By default, all available GPUs are used. To use specific GPUs use `CUDA_VISIBLE_DEVICES`, e.g. `export CUDA_VISIBLE_DEVICES=0,1`

#### Visualization
To visualize intermediate results (optional) run the `visdom.server` in a separate terminal and enable visualization passing the command line arguments: `--visdom True --visdom-port PORT-NUMBER`.

```
# Run the server in a separate terminal
python -m visdom.server -port 8067
```






### Configuration

The available options for each of the provided models ProSR, ProSRs and ProSRGAN are available in the folder `PROJECT_ROOT/options`. Note that the same configuration file is embedded as a dictionary in the respective *.pth. file. You can print the configuration file, as well as the log and evaluation history using the command:

```
python print_info.py --config data/checkpoints/proSR.pth
```




## Testing
Run:
```
python test.py -i LR_INPUT (optional) -t HR_INPUT (optional) --checkpoint CHECKPOINT --upscale-factor NUMBER
```
`LR_INPUT` is the low-resolution input and can be either a folder, an image or a list of images. If high-resolution images are provided (`HR_INPUT`), the script will compute the resulting PSNR and SSIM. Alternatively, if only high-resolution images are given as arguments, the script will scale `HR_INPUT` by the inverse of the upscale factor `NUMBER` and use the result as `LR_INPUT`.

#### Usage
```
# upsample LR_DATA
python test.py -i LR_DATA --checkpoint CHECKPOINT --scale NUMBER

# upsample LR_DATA and evaluate against HR_INPUT
python test.py -i LR_DATA -t HR_INPUT --checkpoint CHECKPOINT --scale NUMBER

# Dowsample HR_INPUT and evaluate upsampled(downsampled(HR_INPUT)) against HR_INPUT
python test.py -t HR_INPUT --checkpoint CHECKPOINT --scale NUMBER
```

`CHECKPOINT` is the path to the pretrained *\*.pth* file.

#### Example
```
# ProSR: Replicate x8 results
python test.py --checkpoint data/checkpoints/proSR.pth --target data/datasets/DIV2K/DIV2K_valid_HR --scale 8

```

See `test.py`
```
usage: test.py [-h] -c CHECKPOINT [-i [INPUT [INPUT ...]]]
               [-t [TARGET [TARGET ...]]] -s SCALE [-f FMT] [-o OUTPUT_DIR]
```

optional arguments:
```
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Checkpoint
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Input images, either list or path to folder. If not
                        given, use bicubically downsampled target image as
                        input
  -t [TARGET [TARGET ...]], --target [TARGET [TARGET ...]]
                        Target images, either list or path to folder
  -s SCALE, --scale SCALE
                        upscale ratio e.g. 2, 4 or 8
  -f FMT, --fmt FMT     Image file format
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output folder.

```
By default, the output images will be saved in `/tmp/<class_name>` where `<class_name>` is the name of the architecture defined in the `checkpoints['params'][class_name]`.

## Additional Tools

### Downscaling
The models available for download have been trained on images downscaled with a bicubic filter. To replicate the same type of downsampling we provide the script `tools/scale.py`:

```
python scale.py -i HR_INPUT -o LR_OUTPUT --ratio 8
```

See `tools/scale.py`
```
usage: scale.py [-h] [-i INPUT] [-o OUTPUT] -s RATIO
```
optional arguments:
```
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image
  -o OUTPUT, --output OUTPUT
                        Output imag.
  -s RATIO, --ratio RATIO
                        scale ratio e.g. 2, 4 or 8

```

### Evaluation
Results can be evaluated in terms of PSNR and SSIM using the script `tools/eval.py`. The command line is similar to `test.py`:

```
python tools/eval.py -i LR_INPUT -t HR_INPUT --scale NUMBER
```

The input can be either a folder, an image or a list of images. The upsampling factor needs to be specified because boundary cropping depends on it.

See `tools/eval.py`:

```
usage: eval.py [-h] -i [INPUT [INPUT ...]] -t [TARGET [TARGET ...]] -u
               UPSCALE_FACTOR
```
optional arguments:
```
  -h, --help            show this help message and exit
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        High-resolution images, either list or path to folder
  -t [TARGET [TARGET ...]], --target [TARGET [TARGET ...]]
                        Super-resolution images, either list or path to folder
  -u UPSCALE_FACTOR, --upscale-factor UPSCALE_FACTOR
                        upscale ratio e.g. 2, 4 or 8
```

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
## Contacts
If you have any question, please contact [Yifan Wang](yifan.wang@inf.ethz.ch) and [Federico Perazzi](fperazzi@adobe.com).
