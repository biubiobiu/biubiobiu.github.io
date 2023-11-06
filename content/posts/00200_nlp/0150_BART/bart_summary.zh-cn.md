---
title: "BART综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: BART综述
    identifier: bart-summary-github
    parent: bart-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["BART"]
categories: ["Basic"]
---


## 一、背景



## 二、BART
<a href="https://arxiv.org/pdf/1910.13461.pdf" target="blank">BART</a> 是一个去噪自动编码器，用于预训练seq-to-seq模型。Bart是标准的Transformer架构，Bart的预训练过程是：

1. 用噪声函数破坏文本
2. 通过学习，让模型重建原始文本。


### 1、模型架构

1. 同GPT一样，把ReLU激活函数修改为GeLU
2. 用 $N(0, 0.02)$ 初始化参数。
3. 基础版：采用6层编码器；large版：采用12层编码器。与bert的差异
    * 解码器的每层，对编码器的最终隐藏层，做交叉attention
    * BERT在进行预测之前，会有一个前馈网络，而BART没有。总体而言，BART的参数比同等大小的BERT模型多了10%

### 2、总结
<a href="https://arxiv.org/pdf/1910.13461.pdf" target="bland">BART</a>提出了各种各样的破坏方法，比如：

* 删掉某些单词(Delete)；
* 打乱输入多个句子的顺序(permutation)； (❌: 效果不好)
* 交换序列中单词的位置(rotation)； (❌: 效果不好)
* 随机插入MASK(比如：原来AB单词之间没有其他单词，故意插入一个MASK去误导模型)或一个MASK盖多个单词(误导模型这里只有一个单词)(Text Infilling)。 (✅: 效果最好)




