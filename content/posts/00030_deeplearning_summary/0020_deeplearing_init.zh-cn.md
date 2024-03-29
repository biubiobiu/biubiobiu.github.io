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

不合适的权重初始化会使得隐藏层数据的方差过大（例如，随着输入数据量的增长，随机初始化的神经元的输出数据的分布中的方差也增大），从而在经过sigmoid这种非线性层时离中心较远(导数接近0)，因此过早地出现梯度消失。所以，在深度学习中，神经网络的权重初始化方法（weight initialization）对模型的收敛速度和性能有着至关重要的影响。一个好的权重初始化虽然不能完全解决梯度消失或梯度爆炸的问题，但是对于处理这两个问题是有很大帮助的，并且十分有利于提升模型的收敛速度和性能表现。<br>

**过大/过小 问题**：
1. 如果权值的初始值过大，则loss function相对于权值参数的梯度值很大，每次利用梯度下降更新参数的时，参数更新的幅度也会很大，这就导致loss function的值在其最小值附近震荡。
2. 而过小的初值值则相反，loss关于权值参数的梯度很小，每次更新参数时，更新的幅度也很小，着就会导致loss的收敛很缓慢，或者在收敛到最小值前在某个局部的极小值收敛了。


> **Glorot条件** ：优秀的初始化应该保证以下两个条件：<br>
> 1. 各个层的激活值h（输出值）的方差要保持一致，
> 2. 各个层对状态z的梯度的方差要保持一致，


> **一个事实：方差与Layer的关系**： <a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf" target="bland">参考</a>
> 1. 各个层激活值h（输出值）的方差与网络的层数有关，<font color=#f00000>激活值的方差逐层递减</font>，就是越来越集中到一定范围，这个范围的概率会较大；
> 2. 关于状态z的梯度的方差与网络的层数有关，<font color=#f00000>状态的梯度在反向传播过程中越往下梯度越小（因为方差越来越小）。</font>；
> 3. 各个层权重参数W的梯度的方差与层数无关；


**初始化的要求**
1. 参数不能全部初始化为0，也不能全部初始化同一个值；
2. 最好保证参数初始化的均值为0，正负交错，正负参数大致上数量相等；
3. 初始化参数不能太大或者是太小，参数太小会导致特征在每层间逐渐缩小而难以产生作用，参数太大会导致数据在逐层间传递时逐渐放大而导致梯度消失发散，不能训练；


### 1、Xavier初始化
Xavier初始化的基本思想：<font color=#f00000>保持输入和输出的方差一致（服从相同的分布）</font>，这样就避免了所有输出值都趋向于0。它为了保证前向传播和反向传播时每一层的方差一致。<br>

在全连接层的Xavier初始化：用 $N(0, 1/m)$ 的随机分布初始化。<br>


### 2、MSRA
Xavier初始化适合用tanh激活函数，对于Relu激活函数比使用。何凯明大神提出了 <a href="https://arxiv.org/pdf/1502.01852.pdf" target="bland">MSRA</a>，可以适用于Relu激活函数。 <br>

主要想要解决的问题是由于经过relu后，方差会发生变化，因此我们初始化权值的方法也应该变化。只考虑输入个数时，MSRA初始化是一个均值为0，方差为2/n的高斯分布：$N(0, \sqrt{2/n})$ 。在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0（x负半轴中是不激活的），所以要保持方差不变，只需要在Xavier的基础上再除以2


### 3、NTK参数化
除了直接用这种方式初始化外，还可以使用 参数化的方式：用 $N(0, 1)$ 的随机分布来初始化，但需要将输出结果除以 $\sqrt{m}$，即：
$$
y_j = b_j + \frac{1}{\sqrt{m}} \sum_i{x_i w_{ij}}
$$

这个高斯过程被称为 "NTK参数化"，可以参考 <a href="https://arxiv.org/abs/1806.07572" target="bland">《Neural Tangent Kernel: Convergence and Generalization in Neural Networks》</a>，<a href="https://arxiv.org/abs/2001.07301" target="bland">《On the infinite width limit of neural networks with a standard parameterization》</a>。利用NTK参数化后，所有参数都可以用方差为1的分布初始化，这意味着每个参数的尺度大致是一个级别，这样的话我们就可以设置较大的学习率，加快收敛。

