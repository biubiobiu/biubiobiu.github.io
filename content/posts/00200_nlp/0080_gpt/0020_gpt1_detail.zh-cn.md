---
title: "GPT-1"
date: 2023-08-08T06:00:20+08:00
menu:
  sidebar:
    name: GPT-1
    identifier: gpt-gpt1-github
    parent: gpt-github
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["GPT-1"]
categories: ["Basic"]
mermaid: true
enableEmoji: true
---

## 一、GPT-1的结构

当时的两个问题：
1. 没有一个合适的目标函数，不同的子任务（比如：翻译、分类、理解等）有自己的目标函数
2. 怎么有效的把学到的 表征 传递到下游的子任务重。因为，NLP的子任务差异还是挺大的。



### 1、输入输出

### 2、编码

对于输入的一句文本串，机器是操作不了的，需要把这段文字串中的每个字转变成<font color=#aa20f0>数字向量</font>。那么如何将单词变成向量呢？
> 1. 构建词表：将所有单词都搜集起来，通过训练一个分词模型，把一句文本split成：单词、固定式语句、组词、等等。比如：GPT的词表大小为 50257。
> 2. one-hot编码：比如：每个单词的one-hot编码，就是一个词表(50257)大小的一个向量，该词位置上的值为1，其余全是0.
> 3. embedding：对于one-hot编码，大部分都是0填充，就是卑鄙的浪费。为了解决这个问题，模型学习了一个embedding函数：一个神经网络，把50257长度的1、0向量，输出n长度的数字向量。即：模型试图将词表映射到较小的空间。(这也比较合理：因为词表中本来就存在：近义词、同义词、等等)

<p align="center"><img src="/datasets/posts/nlp/embedding.png" width="100%" height="100%" title="embedding" alt="embedding"></p>

### 3、位置信息编码(Position Encoding)

文本的位置信息很重要，就想图像中每个像素点的位置信息，不过输入一句话，跟顺序打乱，attention输出都是可以是一样的（如果顺序有变动，相应的权重变动一下就行）。所以，需要手动加入文本的位置信息。位置信息的计算：<br>
> 1. 比如：GPT允许一句输入最长2048个token。每个token经过one-hot编码、embedding后 维度为12288。
> 2. 位置编码的输出是：2048*12288 维的信息。其中，2048方向可以看成时间t(或者离散的n); 12288方向可以看成不同的频率。
> 3. 假设：从1~12288，频率从：$f, ..., f^{12288}$，就可以理解为 $T = 1/f^{12288}$ 进制下的数字表示法。每个位置就是可以是不一样。


<p align="center"><img src="/datasets/posts/nlp/position_em.png" width="50%" height="50%" title="position" alt="position"></p>


### 4、注意力机制

> 1. 文本的embedding + 位置编码，作为注意力机制的输入
> 2. $\bf W_q, W_k, W_v$，三个可学习的矩阵，把输入的embedding向量，变换成向量：$\bf q, k, v$。
> 3. attention计算：用搜索向量$\bf q_i$，与所有key向量$\bf{k_i}$，$i\in(1,..N)$计算内积(表示相似度)。这个N个值分别作为$\bf v_i$ $i \in (1,...,N)$ 的权重。最后计算出的向量就是一个head的attention输出
> 4. 一个head的计算注意力后的向量维度为128，GPT采用96个head，拼接起来正好是12288维度。经过$\bf W_z$ 转换后，作为attention模块的输出，维度与输入一致。

<p align="center"><img src="/datasets/posts/nlp/mha.png" width="100%" height="100%" title="position" alt="position"></p>

### 5、layer normalization



### 6、前馈神经网络


### 7、解码

96个注意力机制/前馈网络 后，输出是是 2048*12288的向量信息。不过词表是50257大小，所以需要把embedding的逆变换，把12288维度映射回50257大小。对下一个字的预测：输出一个50257维的向量，这个向量中的值表示词表中每个字的概率值，通过softmax之后，选出最大概率的字，或者选出top-k个最有可能得词（想象力的体现）。

## 二、


