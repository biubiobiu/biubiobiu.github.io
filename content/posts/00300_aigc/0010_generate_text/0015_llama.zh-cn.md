---
title: "LLaMa"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: LLaMa
    identifier: aigc-text-llama
    parent: aigc-text
    weight: 15
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","llama"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介


## 二、网络结构

### 1、LLaMa

### 2、LLaMa 2

**数据方面**<br>
1. LLaMa2训练了2000B的tokens，训练语料比LLaMa多了40%
    * 2000B 个token的预训练集，提供了良好的性能和成本权衡；对最真实的来源进行上采样，以增加知识并抑制幻觉，保持真实
    * 调查数据，以便用户更好地了解模型的潜在能力和局限性，保证安全。
2. 上下文长度从2048提升到了4096
3. LLaMa2-chat 模型还接受了超过100w的人类标注的训练数据
    * 开源数据选了 LLaMa2
    * 使用监督微调 LLaMa2-chat
    * 使用人类反馈强化学习(RLHF)进行迭代细化；包括拒绝采样、近端策略优化


**网络方面**<br>

<p align="center"><img src="/datasets/posts/nlp/llama_1.png" width=50% height=50%></p>

1. RMSNorm 归一化
2. FFN中用swiGLU激活函数替换原来的Relu
3. 旋转位置编码 RoPE
4. 增加上下文长度
5. 分组查询注意力 GQA
    * 原始的 多头注意力：MHA
    * 具有单个KV投影的原始多查询格式：MQA
    * 具有8个KV投影的分组查询注意力变体：GQA

<p align="center"><img src="/datasets/posts/nlp/GQA_attention.png" width=90% height=90%></p>

**训练方面** <br>

1. 预训练细节：
    * 用AdamW优化器进行训练，其中： $β_1 =0.9，β_2 = 0.95，eps = 10−5$。
    * 使用余弦调整学习率，预热2000steps，$lr$ 衰减到峰值的10%
    * 使用0.1的权重衰减 、1.0的梯度裁剪

2. 精调细节：
    * 余弦学习率，$lr=2e-5$
    * 权重衰减0.1，batch_size=64，序列长度为4096
    * 训练2个epoch
    * 引入Ghost Attention 有助于控制多轮对话
