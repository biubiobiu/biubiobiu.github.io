---
title: 神经网络画图篇
date: 2021-09-09T06:00:20+06:00
menu:
  sidebar:
    name: 神经网络画图篇
    identifier: map-in-deeplearning-github
    parent: deep-learning-summary
    weight: 100
author:
  name: biubiobiu
  image: /images/author/john.png
categories: ["Basic"]
---

## 一、简介

一图抵万言！本篇介绍神经网络的可视化工具和绘图软件。

## 二、示意图

### 1、NN SVG 

提供三种典型的神经网络绘图风格，个性化参数多；交互式绘图。
NN-SVG是由麻省理工学院弗兰克尔生物工程实验室开发的。可以绘制的图包括以节点形式展示的FCNN style，这个特别适合传统的全连接神经网络的绘制。

<a href="https://github.com/alexlenail/NN-SVG" target="blank">Github</a>

<a href="http://alexlenail.me/NN-SVG/" target="blank">Demo</a>



### 2、PlotNeuralNet
底层基于latex的宏指令绘制，上层提供基于python的描述框架，绘制脚本简单。可以绘制复杂的网络结构。

PlotNeuralNet 是由萨尔大学计算机科学专业的一个学生开发的，目前主要支持的是卷积神经网络，其中卷积层、池化层、bottleneck、skip-connection、up-conv、Softmax等常规的层在代码中都有定义，但缺少RNN相关的可视化层展示。

<a href="https://github.com/HarisIqbal88/PlotNeuralNet" target="blank">Github</a>



## 三、计算图

### 1、Netron

Netron是一个神经网络可视化包，支持绝大多数神经网络操作。该功能包可以为不同节点显示不同的颜色，卷积层用蓝色显示，池化层和归一化层用绿色显示，数学操作用黑色显示。在使用方面，可以直接访问网页端，上传模型文件，就可以看到网络结构图，并可以进一步利用pip安装并引入到程序中通过浏览器查看模型的变化。

<a href="https://lutzroeder.github.io/netron/" target="blank">Github</a>

<a href="https://www.lutzroeder.com/ai/netron/" target="blank">Demo</a>

