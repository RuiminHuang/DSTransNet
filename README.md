# DSTransNet

[![Static Badge](https://img.shields.io/badge/building-pass-green?style=flat-square)](https://github.com/RuiminHuang/DSTransNet)
[![Static Badge](https://img.shields.io/badge/language-Python-blue?style=flat-square)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/framework-PyTorch-blue?style=flat-square)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/license-Apache2.0-blue?style=flat-square)](./LICENSE)

The official implementation of the paper "Dynamic Feature Selection: A Novel Network with Feature Enhancement and Dynamic Attention for Infrared Small Target Detection" in PyTorch.

## Abstract

Infrared small target detection (IRSTD) has significantly benefited from UNet-based neural models in recent years. However, current methodologies face challenges in achieving optimal compromise between missed detections and false alarms. To overcome this limitation, we rethink the role of each structural component within UNet-based architectures applied for IRSTD. Accordingly, we conceptualize the UNet's encoder as specializing in feature extraction, the skip connections in feature selection, and the decoder in fusion-based reconstruction. Building upon these conceptualizations, we propose the DSTransNet. Within the feature extraction stage, the edge shape receptive field (ESR) module enhances edge and shape feature extraction and expands the receptive field via multiple convolutional branches, thereby reducing missed detections. At the feature selection stage, the reliable dynamic selection filtering (RDSF) module employs dynamic feature selection, leveraging encoder-based self-attention and decoder-based cross-attention of the Transformer to suppress background features resembling small targets and mitigate false alarms. During the feature fusion-based reconstruction stage, the cross-attention of spaces and channels (CSCE) module emphasizes small target features via spatial and channel cross-attention, reconstructing more accurate multi-scale detection masks. Extensive experiments on the SIRST, NUDT-SIRST, and SIRST-Aug datasets demonstrate that the proposed DSTransNet method outperforms state-of-the-art IRSTD approaches. The code is available at <https://github.com/RuiminHuang/DSTransNet>.

- [Abstract](#abstract)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [File Structure](#file-structure)
  - [Datasets Link](#datasets-link)
- [Training](#training)
- [Test](#test)
- [Model Zoo and Benchmark](#model-zoo-and-benchmark)
  - [Leaderboard](#leaderboard)
  - [Model Zoo](#model-zoo)
- [Citation](#citation)

## Installation

## Dataset Preparation

### File Structure

### Datasets Link

## Training

## Test

## Model Zoo and Benchmark

### Leaderboard

### Model Zoo

## Citation
