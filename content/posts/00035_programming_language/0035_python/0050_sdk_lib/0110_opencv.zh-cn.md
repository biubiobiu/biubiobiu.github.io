---
title: "OpenCV"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: OpenCV
    identifier: python-sdk-opencv
    parent: python-sdk
    weight: 110
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","OpenCV"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

**安装问题**：在环境里安装OpenCV后，在pycharm上没有命令提示。这个可能是OpenCV版本的问题。<br>
**解决方案**：python3 -m pip install --force-reinstall --no-cache -U opencv-python==4.5.5.62



## 一、连通域

1. [x] <font color=#a020f0>cv2.connectedComponentsWithStats</font> <br>
    示例：num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=None)<br>

    <br>
    输入参数：<br>
	* image: 二值图<br>
	* connectivity：可选值为4或者8，表示使用4联通还是8联通<br>
    * ltype：输出图像标记的类型，目前支持CV_32S、CV_16U<br>

    <br>
    输出参数：<br>
    * num_labels: 所有连通域的数目<br>
	* labels：图像上每个像素的标记<br>
	* stats：每个标记的统计信息：是一个5列的矩阵[[x,y,width,height,面积]，]<br>
    * centroids：连通域的中心点<br>

2. [x] <font color=#a020f0>cv2.connectedComponents</font> <br>
    示例：num_objects, labels = cv2.connectedComponents(image)<br>

    <br>
    输入参数：<br>
	* image: 二值图，8bit单通道图像<br>

    <br>
    输出参数：<br>
    * num_labels: 所有连通域的数目<br>



## 二、画图
