---
title: "T5综述"
date: 2023-08-08T06:00:20+08:00
menu:
  sidebar:
    name: T5综述
    identifier: t5-summary
    parent: t5-github
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["T5"]
categories: ["Basic"]
mermaid: true
enableEmoji: true
---

## 一、简介

<a href="https://arxiv.org/pdf/1910.10683.pdf" target="bland">T5</a> <br>
<p align="center"><img src="/datasets/posts/nlp/t5_0.png" width=100% height=100%></p> <br>

> T5的主要贡献是：把NLP的不同任务，统一成一种任务形式：文本输入，文本输出。即：每个任务（包括翻译，问题解答和分类）都将模型文本作为输入，并对其进行训练以生成一些目标文本。这使我们可以在各种任务中使用相同的模型，损失函数，超参数等。“ T5”是指我们的模型，我们将其称为“文本到文本传输转换器”。


> **数据**： “Colossal Clean Crawled Corpus” (C4) 巨大的干净爬行的语料库，这是我们创建的基于常见爬网的数据集，它是未标记文本数据的来源。

## 二、模型介绍
目前基于Transformer的模型架构主要有：
1. Encoder-Decoder结构（传统的Transformer结构）：Seq2Seq 常用模型，在编码器输入中可以看到序列中包括自己的全部字符，在解码器的输出只能看到当前字符及之前的字符。
2. Language model结构（GPT的结构）：Encoder-Decoder结构 的Decoder部分，单向结构，每次只能看到当前以及之前的部分。
3. Prefix LM 结构（UniLM的结构）：前面一部分文本可以看到前缀部分所有内容，后面剩下的内容只能看到自己以及之前的内容。

<p align="center"><img src="/datasets/posts/nlp/T5_1.png" width=70% height=70%></p> <br>

### 1、T5模型结构

通过T5的实验发现Encoder-Decoder结构的模型效果最好，所以T5模型本质上来说是一个基于Transformer的Encoder-Decoder模型。


### 2、位置编码

Transformer的 绝对位置编码，虽然有一定的外推性，但是没有方向性。<br>
T5采用了相对位置编码：根据token与token之间的位置关系来生成权重。比如：$w_{-3}, w_{-2}, w_{-1}, w_{0}, w_1, w_2$ <br>
其中，$w_0$ 表示自己位置的权重，$w_1$ 表示下一个位置的权重

### 3、自监督预训练方法

作者对比了3中训练方法：
1. 语言模型式：单向的从左到右一次预测，就是GPT模型的方式
2. bert-style: 相bert一样随机破坏掉一部分，然后还原
3. 顺序还原：打乱文本的顺序，输出恢复原来顺序。

经过作者实验对比，发现：<font color=#f00000>bert-style 的训练方式，效果最好。</font>

### 4、破坏策略

作者对比3中破坏策略：
1. Mask法：随机破坏一个token，用一个特殊字符替换。
2. Replace Span (小段替换): 将一小段token破坏掉，用一个特殊字符替换。
3. Drop法：没有替换操作，直接丢弃。

经过作者实验对比，发现：<font color=#f00000>Replace Span (小段替换)，效果最好。</font>

破坏比例，作者对比了：$10\\%, 15\\%, 25\\%, 50\\%$，实验发现：<font color=#f00000>破坏 $15\\%$，效果最好。</font>

Span长度，作者对比了：$2, 3, 5, 10$，实验发现：<font color=#f00000>破坏长度 $span = 3$，效果最好。</font>

<p align="center"><img src="/datasets/posts/nlp/T5_2.png" width=90% height=90%></p> <br>



### 5、多任务-微调

大多数的多任务应用是这样的：每个任务可能会有不同的损失函数；每个任务可能用相同的模型训练一遍，也就是说每个任务有自己的checkpoint文件；或者每个任务可能会添加不同的后续网络。<br>
能不能把所有任务的数据放在一起，统一的训练呢？<br>

作者对比了3种混合数据的方式：主要是为了让每个任务的训练数据均衡
1. Examples-proportional mixing：$r_m = \frac{min(e_m, K)}{\sum min(e_m, K)}$，按照这个概率采样。
2. Temperature-scaled mixing：在上面的基础上，对 $r_m^{1/T}$，T 越大，各个任务数据集采样越均衡。
3. Equal mixing：各个任务数据采样概率相同

通过实验发现：
1. <font color=#f00000>多任务的数据混在一起pretrain，再在特定任务上微调 == 无监督 pretrain，再微调</font>
2. 说明：多任务一起训练，是不会相互影响


### 6、模型大小

通过实验发现：<font color=#f00000>增大训练集、增大模型大小、不同任务模型融合</font> 都是可以提升模型性能的。

|模型|Layers|Hidden Size|Attention Head|参数量|
|:--|:--|:--|:--|:--|
|Base|12|768|12|220M|
|Small|6|512|8|60M|
|Large|24|1024|16|770M|
|3B|24|1024|32|3B|
|11B|24|2048|128|11B|


### 总结

T5模型的主要贡献：
1. Text-to-Text：把所有NLP任务都转换为Text-to-Text范式。实验证明：这种范式的性能与特定任务的体系结构性能是有一战之力的，并且在扩大规模后效果更佳
2. Architectures：实验证明：原始的Transformer架构效果最好。在编码器和解码器中共享参数，不会导致性能下降，并且使得总参数数量减半。
3. Unsupervised objectives：训练时 bert-style 的随机破坏方式最好，采用span 小段替换的策略最好
4. C4数据集
5. 训练方式：多任务pretrain + finetune 的性能 与 unsupervised pre-train + fine-tuning 效果差不多
6. Scaling: 使用更多的数据、训练更大的模型、模型融合都能提高性能
7. 提出在各个任务上取得SOTA的T5系列模型

## T5的变体

### 1、mT5

**数据**：
1. 仍然是用C4数据集，但语料不再只限于英语，而是扩大到101中语言

**模型**：
1. 激活函数：由Gate-GELU 替换掉 RELU
2. 无标签数据部做dropout
3. embedding和分类层部做参数共享
4. 更大的d_model，更小的num_heads和d_ff
5. 仅使用C4进行预训练

### 2、byT5

由于tokenizer分词是在一部分数据上训练的，有局限性，如果有错误、缺少等会影响模型性能。<br>
byT5 就直接去掉了tokenizer部分，不需要把自然语言转换为token，而是直接把自然语音作为输入。<br>
在字节序列而不是句子片段子词标记序列上预训练T5模型，避免不同tokenizer分词带来的对语言模型的影响。

### 3、Flan-T5
Flan 是指Instruction fineturn，即：基于指令的微调。Flan-T5是在Flan数据集上训练的T5模型，这些数据集包括：taskmaster2、djaym7/wiki_dialog、deepmind/code_contests、lambada、gsm8k、aqua_rat、esnli、quasc和qed。

