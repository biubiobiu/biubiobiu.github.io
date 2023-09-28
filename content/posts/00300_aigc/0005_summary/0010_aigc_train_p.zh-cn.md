---
title: "大模型训练框架"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 大模型训练框架
    identifier: aigc-summary-train
    parent: aigc-summary
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","大模型", "训练框架"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介


## 二、Deepspeed



## 三、Megatron-LM

<a href="https://arxiv.org/pdf/1909.08053v4.pdf" target="bland">《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》</a> <br>
Megatron 是一篇极具影响力的论文，介绍了高效的模型并行架构。Megatron引入了张量并行(tensor parallelism)，这是一种模型并行的变体，它将模型分割成多块，以实现层内模型并行，从而达到与单个GPU基准线76%效率相当的水平（尽管基准线只有峰值FLOPS的30%）。<br>

Megatron意识到如果，你有一个网络模型 $Y=f(XW)$，你沿着列拆分开了 $W=[W1, W2]$ ，然后 $Y=[f(XW1), f(XW2)]$，所以你不需要做任何操作来同步 $Y$，transformer中唯一需要同步（all-reduce）的点是：
1. 正向传播中，在MLP块后拼接模型激活值之前添加dropout时需要同步。
2. 反向传播中，在self-attention块的开始处需要进行同步。

通过在这两个关键点进行同步操作，可以保证Transformer模型在计算过程中的正确性和一致性。

<p align="center"><img src="/datasets/posts/nlp/Megatron.png" width=60% height=60%></p>

<a href="https://arxiv.org/pdf/2201.11990v3.pdf" target="bland">《Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model》</a> <br>


