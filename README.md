# DSTransNet

[![Static Badge](https://img.shields.io/badge/building-pass-green?style=flat-square)](https://github.com/RuiminHuang/DSTransNet)
[![Static Badge](https://img.shields.io/badge/language-Python-blue?style=flat-square)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/framework-PyTorch-blue?style=flat-square)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/license-Apache2.0-blue?style=flat-square)](./LICENSE)

The official implementation of the paper "Dynamic Feature Selection: A Novel Network with Feature Enhancement and Dynamic Attention for Infrared Small Target Detection" in PyTorch.

> This repository provides **clean and readable** code with **comprehensive test metrics** (Pd, Fa, ROC, and PR curves).

## Contents
- [Introduction](#introduction)
- [The Network](#the-network)
  - [Overall Pipeline](#overall-pipeline)
  - [Core Module](#core-module)
- [:rocket:Installation](#installation)
- [:bar_chart:Dataset Preparation](#dataset-preparation)
  - [Datasets Link](#datasets-link)
  - [File Structure](#file-structure)
- [:fire:Train](#train)
- [:dart:Test](#test)
- [Benchmark and Model Zoo](#benchmark-and-model-zoo)
  - [Quantitative Results](#quantitative-results)
  - [Qualitative Results](#qualitative-results)
  - [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Contact](#contact)


## Introduction

<div align="center">
  <img src="./figures/Figure1.png" width="500" alt="Figure1">
</div>

The heatmaps illustrate the areas of interest for various layers within the model. Layers subjected to specific analysis are indicated by red dots beneath the figure. (a) infrared image. (b) feature extraction stage, heatmap of focused features. (c) feature selection stage. Areas marked by red borders represent the suppression of false-alarm features, and areas marked by green borders signify the enhancement of small target features. (d) feature fusion-based reconstruction stage. Attention is paid to small target features during the reconstruction process. (e) ground truth label.

## The Network

### Overall Pipeline
![Figure2](./figures/Figure2.png)
Overall architecture of the proposed DSTransNet. It incorporates three stages. The feature extraction stage is responsible for finer feature extraction. The feature selection stage aims to suppress false alarms similar to small targets and enhance real small target features. The fusion-based reconstruction stage emphasises small target features and reconstructs the final multi-scale small target mask.

### Core Module
![Figure4](./figures/Figure4.png)
Proposed RDSF module. It serves to suppress false alarms similar to small targets and enhance real small target features.


## Installation

* Step 1. Clone the repository

```shell
git clone git@github.com:RuiminHuang/DSTransNet.git
cd DSTransNet
```

* Step 2. Create environment and install dependencies

```shell
conda create --name DSTransNet python=3.12
conda activate DSTransNet
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tensorboard==2.19.0
pip install tqdm==4.65.0
```


## Dataset Preparation


### Datasets Link
The dataset comes from [this GitHub repository](https://github.com/GrokCV/SeRankDet). The datasets used in this project and the dataset split files can be downloaded from the following links:

* SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1LgnBKcE8Cqlay5GnXfUaLA?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEgG0VEoH3nDbiWDA?e=gkUW2W)
* NUDT-SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/16BbL9H38cIcvaBh4tPNTCw?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEdBMrQDFM1Vi24DQ?e=vBNoN4)
* IRSTD1K Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1nRoZu1eI9BLnpmsxw0Kdwg?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEepi2ipymni0amNQ?e=XZILFh)
* SIRST-AUG Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1_kAocokYSclQNf_ZLWPIhQ?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEfdtbrZhLsbd0ITg?e=thyA6h)


### File Structure

```shell
|- datasets
    |- NUAA
        |-trainval
            |-images
                |-Misc_1.png
                ......
            |-masks
                |-Misc_1.png
                ......
        |-test
            |-images
                |-Misc_50.png
                ......
            |-masks
                |-Misc_50.png
                ......
    |-NUDT
    |-IRSTD1k
    |-SIRSTAUG
```


Before running the code, make sure to update the dataset path in the config file:

![datasets_path](./figures/datasets_path.png)

## Train

```shell
python train.py
```

Before training, specify the target datasets in the configuration file:

![train_config](./figures/train_config.png)

To view train process, run TensorBoard with:

```shell
tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./
```

![train_log](./figures/train_log.png)


## Test

```shell
python test.py
```

Before testing, specify the pre-trained weights and target datasets in the configuration file:

![test_config](./figures/test_config.png)


To view test results, run TensorBoard with:

```shell
tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./
```

![train_log1](./figures/test_log1.png)
![train_log2](./figures/test_log2.png)



## Benchmark and Model Zoo

### Quantitative Results


| Datasets      | mIoU (x10(-2)) | nIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| SIRST         | 75.61  |  75.55 | 98.17 | 17.57 |
| NUDT-SIRST    | 95.03  |  94.84 | 99.53 | 0.46  | 
| IRSTD-1K      | 68.35  |  67.78 | 93.94 | 19.93 |
| SIRST-Aug     | 74.67  |  71.18 | 99.86 | 19.54 |


### Qualitative Results
![Figure8](./figures/Figure8.png)
2D visualization of detection results across different methods on representative images from SIRST and IRSTD1K datasets. Blue, yellow, and red circles denote correct detections, missed detections, and false alarms, respectively.

### Model Zoo

TensorBoard logs, train logs, test logs, pre-trained weights, and test results are available on [Google Drive](https://drive.google.com/drive/folders/1Cktwh19m4gm0PVY63o_HWHXe6CXkqhOf?usp=sharing). Just download and unzip it to the [log path](./logs/).

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```bibtex
@article{huang2025DSTransNet,
  title={Dynamic Feature Selection: A Novel Network with Feature Enhancement and Dynamic Attention for Infrared Small Target Detection},
  journal={}, 
  author={Ruimin, Huang and Jun, Huang and Yong, Ma and Fan, Fan and Yiming, Zhu},
  year={2025},
  volume={},
  number={},
  pages={}
}
```

## Contact

Please feel free to raise issues or email to [huang_ruimin@whu.edu.cn](huang_ruimin@whu.edu.cn) for any questions regarding our DSTransNet.
