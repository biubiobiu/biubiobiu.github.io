---
title: "模型小型化"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 模型小型化
    identifier: aigc-summary-mini
    parent: aigc-summary
    weight: 15
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","大模型", "小型化"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介

目前小型化的方案：
1. 剪枝 Network Pruning
2. 蒸馏 Knowledge Distillation
3. 量化 Parameter Quantization
4. Architecture Design
5. Dynamic Computation


### 1、蒸馏 Knowledge Distillation

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/distillation-0.png" width=100% height=100%></p>
---

{{< /split >}}


### 2、量化 Parameter Quantization


### 3、剪枝 Network Pruning

在权重W中，有些值非常接近于0，这些值好像没有啥作用。说明这些参数是冗余的，可以去掉。


## 二、TensorRT



