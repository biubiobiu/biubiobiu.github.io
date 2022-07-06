---
title: "简介"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 简介
    identifier: cv-image-segment-summary
    parent: cv-image-segment
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["语义分割", "CV"]
categories: ["Basic"]
---

## 一、简介

It's coming soon.


## 二、网络-基于编码器-解码器
### 1、FCN
<a href="https://arxiv.org/abs/1411.4038" target="blank">《Fully Convolutional Networks for Semantic Segmentation》</a>(2015) 
要说语义分割整体实现精度大的跨越还是在FCN（全卷积神经网络）提出之后。它完全改变了之前需要一个窗口来将语义分割任务转变为图片分类任务的观念，FCN完全丢弃了图片分类任务中全连接层，从头到尾都只使用到了卷积层。从FCN后，基于编码器解码器结构的经典网络结构如同雨后春笋般冒了出来

### 2、U-Net
<a href="https://arxiv.org/abs/1505.04597" target="blank">《U-Net: Convolutional Networks for Biomedical Image Segmentation》</a>(2015) 
Unet网络是在医学影像分割中最常用的模型。它的典型特点是，它是U型对称结构，左侧是卷积层，右侧是上采样层（典型的编码器解码器结构）。

另一个特点是，Unet网络的每个卷积层得到的特征图都会concatenate到对应的上采样层，从而实现对每层特征图都有效使用到后续计算中。也就是文中所说的skip-connection。这样，同其他的一些网络结构比如FCN比较，Unet避免了直接在高级feature map中进行监督和loss计算，而是结合了低级feature map中的特征，从而可以使得最终所得到的feature map中既包含了high-level 的feature，也包含很多的low-level的feature，实现了不同scale下feature的融合，提高模型的结果精确度。

### 3、SegNet
<a href="https://arxiv.org/abs/1511.00561" target="blank">《SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation》</a>(2015) 是一个由剑桥大学团队开发的图像分割的开源项目，该项目可以对图像中的物体所在区域进行分割，例如车，马路，行人等，并且精确到像素级别

### 4、Deeplab V1
<a href="https://arxiv.org/abs/1412.7062" target="blank">《Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs》</a>(2015) 
2015 年的ICLR上提出DeepLab V1是结合了深度卷积神经网络（DCNNs）和概率图模型（DenseCRFs）的方法。它将DenseCRFs作为网络的后处理方法。采用DenseCRFs作为后处理的方法，简单来说，就是对一个像素进行分类的时候，不仅考虑DCNN的输出，而且考虑该像素点周围像素点的值，这样语义分割结果边界清楚。


### 5、Deeplab V2
<a href="https://arxiv.org/abs/1606.00915" target="blank">《DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs》</a>(2017) 
在实验中发现 DCNNs 做语义分割时精准度不够的问题，根本原因是重复的池化和下采样降低了分辨率。但是另一方面，重复的池化和下采样扩大了感受野，而感受野的扩大对语义分割任务来说也是至关重要的。针对这一问题，DeepLab v2采用的空洞卷积算法扩展感受野，与此同时不会降低特征图的分辨率。此外，deeplab v2基于空洞卷积，设计了ASPP模块。它组合了不同dilation rate的空洞卷积所产生的特征图。这样，不同空洞卷积产生的不同感受野的特征图被组合在了一起，从而获取了更加丰富的上下文信息。

### 6、PSPnet
<a href="https://arxiv.org/abs/1612.01105" target="blank">《Pyramid Scene Parsing Network》</a>(2017) 


### 7、Deeplab V3
<a href="https://arxiv.org/abs/1706.05587" target="blank">《Rethinking Atrous Convolution for Semantic Image Segmentation》</a>(2017) 
deeplab v3的创新点一是改进了ASPP模块。其实也就是与原来的ASPP相比，新的ASPP模块能够聚集到全局的上下文信息，而之前的只能聚集局部的上下文。

### 8、Deeplab V3+
<a href="https://arxiv.org/abs/1802.02611" target="blank">《Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation》</a>(2018) 

deeplab v3+的创新点：
1. 是设计基于v3的decode module，使得结果变得更加精细；
2. 是用modify xception作为backbone。


## 三、网络-基于注意力

### 1、DANet
<a href="https://arxiv.org/abs/1809.02983" target="blank">《Dual Attention Network for Scene Segmentation》</a>(2019) 

### 2、CCNet
<a href="https://arxiv.org/abs/1811.11721" target="blank">《CCNet: Criss-Cross Attention for Semantic Segmentation》</a>(2019) 

### 3、DANet
<a href="https://arxiv.org/abs/2004.01547" target="blank">《Context Prior for Scene Segmentation》</a>(2020) 

