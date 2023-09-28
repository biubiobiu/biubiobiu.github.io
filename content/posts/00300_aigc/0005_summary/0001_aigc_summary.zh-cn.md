---
title: "综述"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 综述
    identifier: aigc-summary-text
    parent: aigc-summary
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","summary"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

在大语言模型的训练中，如果增大数据量，相应的应该减少学习率，这个跟原来的经验相反。<br>

模型大小与模型效果：<br>
<a href="https://arxiv.org/pdf/2206.07682.pdf" target="bland">《Emergent Abilities of Large Language Models》</a> <br>
这篇文章指出：随着模型大小的增大，模型效果先不会有明显提升；增加到一定程度，模型有个突然顿悟时刻。

<p align="center"><img src="/datasets/posts/nlp/ablity_llm.png" width="90%" height="90%"></p>



## 一、文本生成


### 1、GPT

<a href="/zh-cn/posts/00200_nlp/0080_gpt/0010_gpt_summary">参考</a>

### 2、PaLM
<a href="https://arxiv.org/pdf/2204.02311.pdf" target="bland">《PaLM: Scaling Language Modeling with Pathways》</a> <br>

PaLM才是真正的“大”模型。它是迄今为止训练的最大的密集语言模型，参数为 540B，需要 6144 个 TPU 来训练（这是 3 个完整的 TPU pod，每个包含 2048 个 TPU）。这太贵了！可能只有谷歌拥有资源+基础设施来做到这一点。使用的Token高达7800亿。PaLM是使用Google新一代PathWay分布式训练框架训练出来。<br>

与GPT-3相比的变化：
> 1. 多查询注意力（Multi-query attention）：在每个注意力头中共享K/V（Key/Value）嵌入，但使用独立的Q（Query）嵌入。这样做可以在推理阶段显著提高模型的速度。
> 2. 并行Transformer块：使用并行的Transformer块来提高训练时间，相较于传统的串行设计，可以减少约15%的训练时间。
> 3. SwiGLU激活函数：与GPT-3使用的GELU激活函数不同，这里采用了SwiGLU激活函数。
> 4. 旋转位置编码RoPE嵌入：使用RoPE（Relative Positional Encodings）嵌入代替学习得到的嵌入方式，在长文本上具有更好的性能 。
> 5. 输入-输出嵌入共享：输入和输出embedding矩阵是共享的。
> 6. 无偏置向量：在mlp、normlayer等算法中，都不使用bias，对于大模型，可以提高训练稳定性。
> 7. SentencePiece与256k标记：使用SentencePiece进行分词处理，标记数量为256k。

所以，有很多变化！同样，其中很多都是常见的，例如使用 GPT-3 的学习嵌入向量已经非常过时了，现在几乎没有人这样做。

<p align="center"><img src="/datasets/posts/nlp/palm_0.png" width="90%" height="90%"></p>
<p align="center"><img src="/datasets/posts/nlp/palm_1.png" width="90%" height="90%"></p>



### 3、ChatGLM

Layer Normalization的顺序和残差连接被重新排列，
用于输出标记预测的单个线性层；
ReLU s替换为GELU s
二维位置编码

### 4、BLOOM

使用 ALiBi 位置嵌入，它根据键和查询的距离直接衰减注意力分数。 与原始的 Transformer 和 Rotary 嵌入相比，它可以带来更流畅的训练和更好的下游性能。ALiBi不会在词嵌入中添加位置嵌入；相反，它会使用与其距离成比例的惩罚来偏向查询键的注意力评分。
Embedding Layer Norm 在第一个嵌入层之后立即使用，以避免训练不稳定。
使用了 25 万个标记的词汇表。 使用字节级 BPE。 这样，标记化永远不会产生未知标记
两个全连接层：


### 5、LLaMa

LLaMa结合了PaLM和Chinchilla两个模型的最佳特点，并做出了一些改进：

> 1. 预归一化（Pre-normalize）：在每个Transformer子层之前对输入进行预归一化。
> 2. 使用RMSNorm：使用RMSNorm代替LayerNorm，与Gopher模型中一样。
> 3. SwiGLU激活函数：使用了PaLM中的SwiGLU激活函数，但是维度从PaLM的值改为了新的值。
> 4. 旋转位置嵌入（Rotary positional embeddings）：采用RoPE（相对位置编码）替代了PaLM中的绝对位置嵌入法。
> 5. 使用AdamW：与Chinchilla模型一样，使用AdamW优化算法。

在计算方面的变化有：
> 1. 使用高效的注意力机制（Rabe & Staats, FlashAttention）。
> 2. 梯度检查点（Gradient checkpointing）。

作者唯一的抱怨是他希望他们能够将模型训练更长时间，因为学习曲线与收敛相差甚远！


### 6、Claude

<a href="https://claude.ai/chat/" target="bland">Claude Chat API</a> <br>

### 7、Cohere

### 8、Falcon


### 9、Vicuna


### 10、Guanaco


### 11、MPT


### 12、Lazarus


### 13、WizardLM



## 二、图像生成

### 1、GAN
2014年

### 2、CAN
2017年

### 3、DALL-E
2021年2月<br>
根据文本描述绘画，绘画水平一般。

### 4、CLIP+VQGAN
2021年4月<br>
根据文本描述绘画，绘画水平一般。

### 5、Disco Diffusion
2022年2月<br>
根据文本描述绘画，具有原创性，图片精美，渲染时间长。

### 6、Midjourney
2022年3月<br>
根据文本描述绘画，适合人像，细节突出

### 7、DALL-E2
2022年4月，OpenAI发布DALL-E 2，命名来源于著名画家Dali和机器人总动员Wall-E，是DALL-E的升级版，其分辨率是之前版本的4倍。<br>

DALL-E 2 由三个模型组成：<font color=a00000>CLIP模型、先验模型、扩散模型</font>。
1. CLIP模型主要是用来对齐文本和图像特征：获取文本编码
2. 先验模型主要是将文本表征映射为图片表征：将文本编码映射为图片编码
3. 扩散模型是根据图片表征来完成完整的图像：用图片编码生成完整的图片。

根据文本描述绘画，限制较多，对复杂文字理解准确，渲染快

### 8、Stable Diffusion
2022年8月，慕尼黑大学的Robin Rombach和Patrick Esser的团队提出的文本生成图像模型，交互简单，生成速度快。Stable Diffusion主要由三部分组成，分别是 <font color=#a00000>VAE、U-Net、CLIP文本编码器</font>：
1. 首先使用CLIP模型将文本转换为表征形式
2. 然后引导扩散模型U-Net在低维表征上进行扩散
3. 最后将扩散后的低维表征送入VAE中的解码器，从而生成图像。

在GAN和CLIP的基础上，Stable Diffusion模型开源，直接推动了AIGC技术的突破性发展。<br>
Stable Diffusion 扩散模型的原理是：先添加噪声后降噪。即：给现有的图像逐步添加噪声，直到图像被完全破坏，然后根据给定的高斯噪声，逆向逐步还原出原图。在模型训练完毕后，只需要输入一段随机的高斯噪声，就能生成一张图像。<br>
根据文本描述绘画，具有原创性，灵活度高，图片精美，具有真实感，渲染快。<br>



### 9、Imagen
2022年11月<br>
优先开源，效果好于DALL-E


## 三、国内

### 1、太极

腾讯基于自身在自然语言处理和图像多模态等方面积累的经验，打造了通用场景模型——<font color=#f00000>太极文生图大模型</font>。太极文生图采用了Diffusion路线

### 2、文心一格
百度提出的AIGC大模型——<font color=#f00000>ERNIE-ViLG 文生图模型</font>，包括：工业设计、游戏制作、服装设计、Logo设计、盆栽设计、动漫设计、珠宝设计、传统艺术等领域。ERNIE-ViLG模型能够深刻地理解中文语境，更了解中国化。

### 3、太乙
IDEA研究院开源的第一个中文版Stable Diffusion模型——<font color=#f00000>太乙 Stable Diffusion</font>，该模型基于0.2亿筛选过的中文图文对进行训练，从而实现了具备中文内核的AIGC模型。

### 4、CogView

智源研究院于2022年上半年，推出的CogView 2.0和 CogVideo

### 5、MSRA

2021年11月微软亚洲研究院与北京大学联合发布了女娲模型，女娲模型用来从输入的文本、图像、视频中生成图像或者视频。

### 6、MagicMix

字节跳动公司发布了MagicMix模型，模型可以将任意两个语义进行组合，生成全新的概念，再基于新概念进行图像生成。

### 7、DPM-Solver

清华大学的朱军教授团队提出的DPM-Solver，是一种针对扩散模型特殊设计的高效求解器。
