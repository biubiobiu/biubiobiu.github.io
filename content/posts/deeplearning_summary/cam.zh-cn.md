---
title: CAM
date: 2022-09-09T06:00:20+06:00
menu:
  sidebar:
    name: CAM
    identifier: cam-github
    parent: deep-learning-summary
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["vlp","summary"]
categories: ["Basic"]
math: true
---

## 一、简介


## 二、模型


### 1、gradient-based

#### 1. GAP
<a href="https://arxiv.org/abs/1512.04150" target="blank">《Learning Deep Features for Discriminative Localizatiion》</a> 

```python
# 代码非常简单， 提取到特征图和目标类别全连接的权重，直接加权求和，再经过relu操作去除负值，最后归一化获取CAM，具体如下:
# 获取全连接层的权重
self._fc_weights = self.model._modules.get(fc_layer).weight.data
# 获取目标类别的权重作为特征权重
weights=self._fc_weights[class_idx, :]
# 这里self.hook_a为最后一层特征图的输出
batch_cams = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
# relu操作,去除负值
batch_cams = F.relu(batch_cams, inplace=True)
# 归一化操作
batch_cams = self._normalize(batch_cams)

```

#### 2. Grad-CAM
<a href="https://arxiv.org/abs/1610.02391" target="blank">《Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization》</a> 

```python


```


### 2、gradient-free







