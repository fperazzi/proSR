# ProSR

Unofficial implementation of [A Fully Progressive Approach to Single-Image Super-Resolution](https://fperazzi.github.io/files/publications/prosr.pdf).

![](data/prosr-teaser.jpg)

ProSR is a Single Image Super-Resolution (SISR) method designed upon the principle of multi-scale progressiveness. The architecture resembles an asymmetric pyramidal structure with more layers in the upper levels to enable high upsampling ratios while remaining efficient. The training procedure implements the paradigm of curriculum learning by gradually increasing the difficulty of the task.
![](data/prosr-arch.jpg)

## Gettings Started
Follow these intrunctions blow to get this code up and running on your local machine for developement and testing purposes.

### System Requirements
*ProSR* is developed under Ubuntu 16.04 with CUDA 9.1, cuDNN v7.0 and pytorch-0.4.0.
We tested the program on Nvidia Titan X and Tesla K40c GPUs. Any NVIDIA GPU with ~12GB memory will do. Parallel processing on multiple GPUs will be supported during training.

### Dependencies
  * Python 3.x
  * pytorch 0.4.0
  * See the full list of dependencies in `PROJECT_ROOT/conda-deps.yml`.

Dependencies can be installed in a conda enviroment executing:
> `conda create --name proSR --file PROJECT_ROOT/conda-deps.yml`

Include `proSR` into the search path setting `export PYTHONPATH=$PROJECT_ROOT/lib:$PYTHONPATH`.


## Training
Not implemented yet. Send an email to [fperazzi@adobe.com](fpearzzi@adobe.com) if you want to be notified when available.

### Configuration
The available options are defined in `lib/prosr/config.py`.

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


## Data
### Pretrained Models
In `PROJECT_ROOT/data` we provide a script `get_data.sh` to download the a pretrained model for x8 upsampling. This model was trained on the [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) dataset. On the "Track 1" of the [NTIRE Super-Resolution Challenge 2018](https://competitions.codalab.org/competitions/18015), it ranked 4th place in terms of PSNR and second when measured with SSIM.

TLDR; Download the data: `sh data/get_data.sh`

### Results
Coming soon...

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


