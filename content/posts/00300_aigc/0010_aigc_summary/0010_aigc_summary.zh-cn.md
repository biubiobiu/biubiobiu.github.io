---
title: "简介"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 简介
    identifier: aigc-summary-1
    parent: aigc-summary
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","summary"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、LLM

### 1、LLaMa

使用RMSNorm（即Root Mean square Layer Normalization）对输入数据进行标准化，RMSNorm可以参考论文：Root mean square layer normalization。
使用激活函数SwiGLU， 该函数可以参考PALM论文：Glu variants improve transformer。
使用Rotary Embeddings进行位置编码，该编码可以参考论文 Roformer: Enhanced transformer with rotary position embedding。
使用了AdamW优化器，并使用cosine learning rate schedule，
使用因果多头注意的有效实现来减少内存使用和运行时间。该实现可在xformers


### 2、PaLM

采用SwiGLU激活函数：用于 MLP 中间激活，采用SwiGLU激活函数：用于 MLP 中间激活，因为与标准 ReLU、GELU 或 Swish 激活相比，《GLU Variants Improve Transformer》论文里提到：SwiGLU 已被证明可以显著提高模型效果
提出Parallel Layers：每个 Transformer 结构中的“并行”公式：与 GPT-J-6B 中一样，使用的是标准“序列化”公式。并行公式使大规模训练速度提高了大约 15%。消融实验显示在 8B 参数量下模型效果下降很小，但在 62B 参数量下没有模型效果下降的现象。
Multi-Query Attention：每个头共享键/值的映射，即“key”和“value”被投影到 [1, h]，但“query”仍被投影到形状 [k, h]，这种操作对模型质量和训练速度没有影响，但在自回归解码时间上有效节省了成本。
使用RoPE embeddings：使用的不是绝对或相对位置嵌入，而是RoPE，是因为 RoPE 嵌入在长文本上具有更好的性能 ，
采用Shared Input-Output Embeddings:输入和输出embedding矩阵是共享的，这个我理解类似于word2vec的输入W和输出W'：

### 3、GLM

Layer Normalization的顺序和残差连接被重新排列，
用于输出标记预测的单个线性层；
ReLU s替换为GELU s
二维位置编码

### 4、BLOOM

使用 ALiBi 位置嵌入，它根据键和查询的距离直接衰减注意力分数。 与原始的 Transformer 和 Rotary 嵌入相比，它可以带来更流畅的训练和更好的下游性能。ALiBi不会在词嵌入中添加位置嵌入；相反，它会使用与其距离成比例的惩罚来偏向查询键的注意力评分。
Embedding Layer Norm 在第一个嵌入层之后立即使用，以避免训练不稳定。
使用了 25 万个标记的词汇表。 使用字节级 BPE。 这样，标记化永远不会产生未知标记
两个全连接层：


### 5、GPT

GPT 使用 Transformer 的 Decoder 结构，并对 Transformer Decoder 进行了一些改动，原本的 Decoder 包含了两个 Multi-Head Attention 结构，GPT 只保留了 Mask Multi-Head Attention，如下图所示:


