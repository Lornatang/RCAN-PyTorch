# RCAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758).

### Table of contents

- [RCAN-PyTorch](#rcan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution Using Very Deep Residual Channel Attention Networks](#about-image-super-resolution-using-very-deep-residual-channel-attention-networks)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download valid dataset](#download-valid-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](#image-super-resolution-using-very-deep-residual-channel-attention-networks)

## About Image Super-Resolution Using Very Deep Residual Channel Attention Networks

If you're new to RCAN, here's an abstract straight from the paper:

Convolutional neural network (CNN) depth is of crucial importance for image super-resolution (SR). However, we observe that deeper networks for image
SR are more difficult to train. The lowresolution inputs and features contain abundant low-frequency information, which is treated equally across
channels, hence hindering the representational ability of CNNs. To solve these problems, we propose the very deep residual channel attention
networks (RCAN). Specifically, we propose a residual in residual (RIR) structure to form very deep network, which consists of several residual groups
with long skip connections. Each residual group contains some residual blocks with short skip connections. Meanwhile, RIR allows abundant
low-frequency information to be bypassed through multiple skip connections, making the main network focus on learning high-frequency information.
Furthermore, we propose a channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels.
Extensive experiments show that our RCAN achieves better accuracy and visual improvements against state-of-the-art methods.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

### Download train dataset

#### DIV2K

- Image format
    - [Baidu Driver](https://pan.baidu.com/s/1EXXbhxxRDtqPosT2WL8NkA) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GtQuoEN78q3AIP8vkh-17X90thYp_FfU/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1dlPcpwRPUBOnxlfW5--S5g) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1KBS38UAjM7bJ_e6a54eHaA) access:`llot`

#### BSD200

- Image format
    - [Google Driver](https://drive.google.com/file/d/1cdMYTPr77RdOgyAvJPMQqaJHWrD5ma5n/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1xahPw4dNNc3XspMMOuw1Bw) access:`llot`

## Test

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to valid mode.
- line 65: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 41: `resume` change to `True`.
- line 42: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 43: `start_epoch` change number of training iterations in the previous round.
- line 44: `resume_weight` the weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1807.02758.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |     PSNR     |
|:-------:|:-----:|:------------:|
|  Set5   |   2   | 38.27(**-**) |
|  Set5   |   3   | 34.74(**-**) |
|  Set5   |   4   | 32.63(**-**) |
|  Set5   |   8   | 27.31(**-**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Image Super-Resolution Using Very Deep Residual Channel Attention Networks

_Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu_ <br>

**Abstract** <br>
Convolutional neural network (CNN) depth is of crucial importance for image super-resolution (SR). However, we observe that deeper networks for image
SR are more difficult to train. The low-resolution inputs and features contain abundant low-frequency information, which is treated equally across
channels, hence hindering the representational ability of CNNs. To solve these problems, we propose the very deep residual channel attention
networks (RCAN). Specifically, we propose a residual in residual (RIR) structure to form very deep network, which consists of several residual groups
with long skip connections. Each residual group contains some residual blocks with short skip connections. Meanwhile, RIR allows abundant
low-frequency information to be bypassed through multiple skip connections, making the main network focus on learning high-frequency information.
Furthermore, we propose a channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels.
Extensive experiments show that our RCAN achieves better accuracy and visual improvements against state-of-the-art methods.

[[Code]](https://github.com/yulunzhang/RCAN) [[Paper]](https://arxiv.org/pdf/1807.02758)

```
@article{DBLP:journals/corr/abs-1807-02758,
  author    = {Yulun Zhang and
               Kunpeng Li and
               Kai Li and
               Lichen Wang and
               Bineng Zhong and
               Yun Fu},
  title     = {Image Super-Resolution Using Very Deep Residual Channel Attention
               Networks},
  journal   = {CoRR},
  volume    = {abs/1807.02758},
  year      = {2018},
  url       = {http://arxiv.org/abs/1807.02758},
  eprinttype = {arXiv},
  eprint    = {1807.02758},
  timestamp = {Tue, 20 Nov 2018 12:24:39 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1807-02758.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
