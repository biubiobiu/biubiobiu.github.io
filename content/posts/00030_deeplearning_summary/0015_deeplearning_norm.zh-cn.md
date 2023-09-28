---
title: "归一化"
date: 2021-08-05T12:30:40+08:00
description: Markdown rendering samples
menu:
  sidebar:
    name: 归一化
    identifier: deep-learning-normal
    parent: deep-learning-summary
    weight: 15
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["机器学习","深度学习","归一化"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、Normlization介绍

一般而言，样本特征由于来源及度量单位不同，其尺度往往差异很大。如果尺度差异很大，神经网络就比较难训练。为了提高训练效率，对输入特征做归一化，把不同的尺度压缩到一定范围内，尺度统一后，大部分位置的梯度方向近似于最优解搜索方向。这样，在用梯度下降法进行求解时，每一步梯度的方向都基本上指向最小值，训练效率会大大提高。<br>

> 归一化：泛指把数据特征转换为相同尺度的方法，比如：
>  1. 把数据特征映射到 [0, 1] 或者 [-1, 1] 区间
>  2. 映射为服从 N(0, 1) 的标准正态分布

### 1、逐层归一化
逐层归一化可以有效提高训练效率的原因：
1. 更好的尺度不变形<br>
深度神经网路中，一个神经层的输入是之前神经层的输出。给定一个神经层 $l$，它之前的神经层 $1, 2, ..., l-1$，的参数变化会导致其输入的分布发生很大的变化。当使用随机梯度下降法训练网络时，每次参数更新都会导致该神经层的输入分布发生变化，层数越高，其输入分布会改变得越明显。<br>
为了缓解这个问题，可以对每个神经层的输入进行归一化，使其分布保持稳定。不管底层的参数如何变化，高层的输入相对稳定。另外，尺度不变性，可以使我们更加高效地进行参数初始化以及超参数选择。<br>

2. 更平滑的优化地形<br>
逐层归一化，一方面可以是大部分神经层的输入处于不饱和区域，从而让梯度变大，避免梯度消失问题；另一方面可以使得神经网络的优化地形（Optimization Landscape）更加平滑，并使梯度变得更加稳定，从而允许使用更高的学习率，并加快收敛速度。

#### 1. 批量归一化

批量归一化（Batch Normalization）对神经网络中的任意中间层进行归一化。<br>

$$
a^{(l)} = f(z^{(l)}) = f(Wa^{(l-1)}+b)
$$
$f(·)$ 是激活函数，$W$ 和 $b$ 是可学习的参数。<br>
为了提高优化效率，就要使净输入 $z^l$ 的分布一致，比如：都归一化为标准正态分布。虽然归一化操作可以应用在输入 $a^{(l-1)}$ 上，但归一化 $z^l$ 更加有利于优化。因此，在实践中，归一化操作一般应用在<font color=#a00000>仿射变换之后，激活函数之前</font>。

#### 2. 层归一化
层归一化（Layer Normalization）是和批量归一化非常类似的方法，与批量归一化不同的是，层归一化是对一个中间层的所有神经元进行归一化。


## 二、Norm的位置

在目前大模型中 Normalization 的位置：
<p align="center"><img src="/datasets/posts/nlp/pre_normalization.jpg" width=90% height=90%></p>

<p align="center"><img src="/datasets/posts/dp_summary/norm_pre_post.jpg" width=90% height=90%></p>

pre Norm 的状态： $x_{t+1} = x_t + F_t(Norm(x_t))$

post Norm 的状态： $x_{t+1} = Norm(x_t + F_t(x_t))$


### 1、为什么Pre效果弱于post

<a href="https://kexue.fm/archives/9009" target="bland">来自苏神的解释</a>: pre-Norm的深度有水分，到一定深度后增加网络深度的效果等同于增加宽度，而不是深度。 <br>

**Pre-Norm**
$$
\begin{aligned}
x_{t+1} &= x_t + F_t(Norm(x_t)) \\\
&= x_{t-1} + F_{t-1}(Norm(x_{t-1})) + F_t(Norm(x_t)) \\\
&= ... \\\
&=x_0 + F_0(Norm(x_0)) + ... + F_{t-1}(Norm(x_{t-1})) + F_t(Norm(x_t))
\end{aligned}
$$

当 $t$ 比较大时，$F_{t-1}(Norm(x_{t-1}))$ 与 $F_t(Norm(x_t))$ 很接近，等效于一个更宽的 $t$ 层模型，所以，在Pre-Norm中多层叠加的结果更多的是增加宽度而不是深度，深度上有水分。<font color=#f00000>在模型训练中，深度通常比宽度更重要。</font><br>

**Post-Norm**<br>

回顾一下残差链接：$x_{t+1} = x_t + F_t(x_t)$<br>
由于残差分支的存在，$x_{t+1}$的方差是 $x_t$ 与 $F_t(x_t)$ 之和 $\sigma^2_1 + \sigma^2_2$，<font color=#f00000>残差会进一步放大方差</font> 。所以需要缩小方差，其中Normalization就可以实现。在Norm过程中方差的变化：

$$
\begin{aligned}
x_l &= \frac{x_{l-1}}{2^{1/2}} + \frac{F_{l-1}(x_{l-1})}{2^{1/2}} \\\
&=... \\\
&= \frac{x_0}{2^{l/2}} + \frac{F_0(x_0)}{2^{l/2}} + \frac{F_1(x_1)}{2^{(l-1)/2}} + ... + \frac{F_{l-1}(x_{l-1})}{2^{1/2}} 
\end{aligned}
$$

在每条残差通道上都有权重缩小，距离越远的削弱的越严重，原始残差效果越来越弱，因此还是不容易训练。所以，post-Norm 通常要warmup+较小的学习率才能收敛。相关分析见：<a href="https://arxiv.org/abs/2002.04745" target="bland">《On Layer Normalization in the Transformer Architecture》</a> <br>

苏神认为：在初始阶段保持一个恒等式，即：引入一个初始化0的参数 $\alpha_t$。从0开始以固定的、很小的步长慢慢递增，直到增加到 $\alpha_t = 1$就固定下来。苏神在实验结果中，这种更新模式获得了最优的结果。
$$
x_{t+1} = x_t + \alpha_t F_t(x_t)
$$


### 2、RMSNorm为啥有效

**Layer Normalizaton** <br>

$$
y = \frac{x-E(x)}{\sqrt{Var(x) + \varepsilon}}
$$


**RMSNorm** <br>

$$
\bar{a_i} = \frac{a_i}{\sqrt{\frac{1}{n} \sum^n_{i=1}a^2_i}} g_i
$$

<a href="https://arxiv.org/abs/1910.07467" target="bland">《Root Mean Square Layer Normalization》</a> <br>

直接去掉了 $E(x)$ 的计算，相当于只是把分布的方差变成了1，中心不一定在0。<a href="https://arxiv.org/abs/2102.11972" target="bland">《Do Transformer Modifications Transfer Across Implementations and Applications?》</a> 这篇文章做了比较充分的对比实验，显示RMSNorm的优越性。<br>

同样在这篇论文中
<a href="https://arxiv.org/abs/1912.04958" target="bland">《Analyzing and Improving the Image Quality of StyleGAN》</a> 提出了一个问题：<br>
在StyleGAN2里，发现所用的 Instance Normalization会导致部分生成图片出现 “水珠”，他们 最终去掉了 Instance Normalization并换用了一个叫 Weight demodulation 的东西。但他们发现如果保留 Instance Normalization 但去掉 中心 $E(x)$ 操作，也能改善这种现象。这也能佐证 中心 $E(x)$ 操作可能会带来负面效果。<br>

一种直观的猜测：中心 $E(x)$ 类似于全连接层的 bias，存储的是关于预训练任务的一种先验分布信息，而把这种先验分布信息直接存储在模型中，反而可能会导致模型的迁移能力下降。随意 $T5$ 不仅去掉了Layer Normalization的 中心 $E(x)$，也把每一层的bias项去掉了。
 