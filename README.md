# A Fully Progressive Approach to Single-Image Super-Resolution

Unofficial Implementation of CVPRW 2018 [NTIRE](http://www.vision.ee.ethz.ch/ntire18/) workshop paper "A Fully Progressive Approach to Single-Image Super-Resolution".

>
> @InProceedings{Wang_2018_CVPR_Workshops,
  author = {Wang, Yifan and Perazzi, Federico and McWilliams, Brian and Sorkine-Hornung, Alexander and Sorkine-Hornung, Olga and Schroers, Christopher},
  title = {A Fully Progressive Approach to Single-Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}

## Usage
### Installation
*ProSR* is developed under Ubuntu 16.04 with CUDA 9.1 and cuDNN v7.0 and pytorch-0.3.1.
We tested the program on Nvidia Titan X and Tesla K40c GPU. Parallel processing on multiple GPU is supported during training.

We provide the full list of dependencies in `PROJECT_ROOT/conda-deps.yml`. They can be installed in a conda enviroment executing:
`conda create --name proSR --file PROJECT_ROOT/conda-deps.yml`

### Overview
The options are defined in `lib/prosr/config.py`.

### Data
We use DIV2K training data. It can be downloaded [here](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).

Validation data includes:
[DIV2K_val](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip), [URBAN_100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)


### Training
Coming soon...

### Testing
 `python test.py -i <list-of-images> -w <model-parameters.pth> -o <output-dir>`

Output images will be saved in the same folder (postfixed with "_proSR") if `--output-dir` is left undefined.

