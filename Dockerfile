FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN conda update -n base conda

# Install image libraries
RUN conda install scikit-image cython

# Install visdom
RUN conda install visdom dominate -c conda-forge

RUN pip install easydict
RUN apt-get update \
  && apt-get install -y unzip \
  && apt-get clean

WORKDIR /proSR

# Download the pretrained models from https://www.dropbox.com/s/3fjp5dd70wuuixl/proSR.zip?dl=0
RUN mkdir data \
  && curl "https://uc3b1ac4fd89b8faade327413784.dl.dropboxusercontent.com/cd/0/get/Asc95ggr2ND1PSdArswBOAHZSVD81uKKJUSh4DkwnFEsotIY0GxBQV6u5Qk2qq9MzMf_LRwpbSqUwLjdS3e7cJBiCamP2GfEJLYaj1IFbCuY8A/file?_download_id=4106874430186797740972858684583935099590670666545264828231381385131&_notify_domain=www.dropbox.com&dl=1" > data/proSR.zip \
  && unzip -d data data/proSR.zip \
  && rm data/proSR.zip
  
COPY . .

ENV PYTHONPATH=/proSR/lib:$PYTHONPATH

