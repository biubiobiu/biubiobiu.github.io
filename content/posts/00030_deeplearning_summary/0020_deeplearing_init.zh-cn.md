---
title: "初始化"
date: 2023-08-05T12:30:40+08:00
description: Markdown rendering samples
menu:
  sidebar:
    name: 初始化
    identifier: deep-learning-init
    parent: deep-learning-summary
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["机器学习","深度学习","初始化"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、初始化

### 1、Xavier初始化
在全连接层的Xavier初始化：用 $N(0, 1/m)$ 的随机分布初始化。<br>


### 2、NTK参数化
除了直接用这种方式初始化外，还可以使用 参数化的方式：用 $N(0, 1)$ 的随机分布来初始化，但需要将输出结果除以 $\sqrt{m}$，即：
$$
y_j = b_j + \frac{1}{\sqrt{m}} \sum_i{x_i w_{ij}}
$$

这个高斯过程被称为 "NTK参数化"，可以参考 <a href="https://arxiv.org/abs/1806.07572" target="bland">《Neural Tangent Kernel: Convergence and Generalization in Neural Networks》</a>，<a href="https://arxiv.org/abs/2001.07301" target="bland">《On the infinite width limit of neural networks with a standard parameterization》</a>。利用NTK参数化后，所有参数都可以用方差为1的分布初始化，这意味着每个参数的尺度大致是一个级别，这样的话我们就可以设置较大的学习率，加快收敛。

