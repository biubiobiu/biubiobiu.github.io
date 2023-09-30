---
title: "混合精度训练"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 混合精度训练
    identifier: aigc-summary-amp
    parent: aigc-summary
    weight: 12
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","大模型", "混合精度训练"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介

目前，混合精度 (Automatically Mixed Precision, AMP) 训练已经成为了炼丹师的标配工具，仅仅只需几行代码，就能让显存占用减半，训练速度加倍。 <br>
AMP 技术是由百度和 NIVDIA 团队在 2017 年提出的 (<a href="https://arxiv.org/pdf/1710.03740.pdf" target="bland">Mixed Precision Training</a>)，该成果发表在 ICLR 上。PyTorch 1.6之前，大家都是用 NVIDIA 的 apex 库来实现 AMP 训练。1.6 版本之后，PyTorch 出厂自带 AMP。

```python
# 原代码
output = net(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# 使用混合精度训练
with torch.cuda.amp.autocast():
    output = net(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

**半精度浮点数 (FP16)**： 是一种计算机使用的二进制浮点数数据类型，使用 2 字节 (16 位) 存储。而 PyTorch 默认使用 **单精度浮点数 (FP32)** 来进行网络模型的计算和权重存储。FP32 在内存中用 4 字节 (32 位) 存储。<br>


## 二、训练流程

<p align="center"><img src="/datasets/posts/nlp/mix_precision_0.png" width=90% height=90%></p>

**训练流程**：
1. 把<font color=#f00000数据输入、模型参数</a>都转换为FP16，forward的计算结果也是FP16，梯度的计算也是FP16
2. 同时维护一份 FP32 的模型权重副本用于更新
3. 在backward阶段，梯度是FP16存储的，在更新到FP32模型权重前，先把梯度的FP16，转换为FP32

**混合精度的问题**：<br>
把模型权重和输入从 FP32 转化成 FP16，虽然速度可以翻倍，但是模型的精度会被严重影响。FP16 的表示范围不大，会出现<font color=#f00000>上/下溢</font>。
