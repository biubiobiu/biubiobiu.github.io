---
title: "ChatGLM"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: ChatGLM
    identifier: aigc-text-chatglm
    parent: aigc-text
    weight: 25
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","ChatGLM"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介

<a href="https://arxiv.org/pdf/2103.10360.pdf" target="bland">《GLM: General Language Model Pretraining with Autoregressive Blank Infilling》</a> <br>

<a href="https://github.com/THUDM/ChatGLM-6B" target="bland">参考</a> <br>

**ChatGLM-6B**：
ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。<br>

**ChatGLM2-6B**：
ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：

1. **更强大的性能**：
    * 基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数
    * 经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
2. **更长的上下文**：基于 <a href="https://github.com/Dao-AILab/flash-attention" target="bland">FlashAttention</a> 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。
3. **更高效的推理**：基于 <a href="https://arxiv.org/pdf/1911.02150.pdf" target="bland">Multi-Query Attention</a> 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。


## 二、网络结构

按照自动编码的思想从输入文本中随机删除连续的标记span，并按照自回归预训练的思想训练模型顺序重建span。<br>
<p align="center"><img src="/datasets/posts/nlp/glm_span.png" width="90%" height="90%"></p>


1. 给定输入 $\bold x = [x_1, x_2,..., x_n]$，采样多个文本span，即： $[\bold s_1, \bold s_2, ..., \bold s_m]$。每个span是输入中一段连续tokens。
    * 调整 span 的长度和数量，就可以让模型变成 自编码(Bert) 或者 自回归(GPT)。比如：让 $span == input$，就表示是没有自编码(Bert)，只有 自回归(GPT)；$ len(span) == 1 $，就是类似bert。
2. 每个span被单个 $[mask]$ 替换，形成损坏的文本，即：图中 $Part A$；打散 $[\bold s_1, \bold s_2, ..., \bold s_m]$ 的顺序，得到 图中 $Part B$。这种排序方式有多种，设：数量有m个，排序的集合： $\bold{Z_m}$，模型会用集合中所有的顺序来做训练，<font color=#f00000>这就相当于数据增广</font>，类似于排列语言模型(XLNet)，与XLNet不同的是：GLM是不会感知 [mask] 的原长度。
3. 两个维度的位置编码：
    * Position 1: 在 已损坏的文本 即：$Part A$ 部分，中的位置id
    * Position 2: 在 $Part B$ 部分，中的位置id
3. 模型以自监督的方式预测span中的tokens，
    * 在 $Part A$ 部分，图d中蓝色部分，类似Bert的MLM，能看到整个蓝色部分。
    * 在 $Part B$ 部分，图d中黄色部分、绿色部分，类似GPT的自监督，能看到 蓝色部分的 $Part A$ + 已经预测的 $Part B$ 部分。
4. 特殊标记 [START] 和 [END] 填充 $Part B$，[START] 表示新的span的输入开始；[END] 表示当前span的输出结束。通过这种方式，我们的模型在统一模型中自动学习双向编码器（对于 A 部分）和单向解码器（对于 B 部分）
5. 多任务：希望同时兼顾 文本理解(NLU) 和 文本生成(NLG)。<font color=#f00000>目标函数式一样的，只是 span的数量和长度</font>。
    * 文本生成(NLG) - 文件级span：在采样span时，长度为输入文本的 $50\\% - 100\\%$，由均匀分布随机产生，这样就更偏向于 文本生成(NLG)
    * 文本理解(NLU) - 句子级span：span 的长度，长度为输入文本的 $15\\%$，由 $\lambda=3$ 的泊松分布随机产生，旨在 内容理解(NLU)

$$
max_\theta E_{z \in Z_m} [\sum^m_{i=1}log p_\theta(\bold{s_{z_i}}|\bold{x_损}, \bold s_{z_{<i}})]
$$


**模型结构**：<br>
1. DeepNorm：稳定训练1000层的Post-Norm方法。微软的 <a href="https://arxiv.org/pdf/2203.00555.pdf" target="bland">DeepNet</a> 这篇指出，因为模型在训练前期，增量更新过大，导致优化到了不好的局部最优点，通过调整残差、更改初始化，稳定了模型前期的前期增量。<br> $ DeepNorm(x) = LayerNorm(a·x + g(x))$
2. 旋转位置编码：RoPE
3. 门控注意单元(GLU)：FFN层的替换，稳定提升模型性能。<br> $FFN_{GLU}(x, \bold{W}, \bold{V}, \bold{W_2}) = (\sigma (xW_1)\otimes{x \bold{V}})\bold{W_2}$


**稳定训练**：<br>
比如：GPT-3模型需要2.8TB显存来存放训练状态、中间激活函数值。<br>
<p align="center"><img src="/datasets/posts/nlp/gpt3_p_0.png" width=70% height=70%></p>
如果直接训练，肯定是不肯能的，需要一些并行优化策略：

1. ZeRO优化器，在数据并行组内分摊优化器状态
2. 使用模型并行，将模型参数分布到多个显卡上。并行策略：
    * 张量并行，随着模型规模增大缓慢扩展，但不超过单机规模
    * 其余全部使用流水线并行，通过调整微批处理大小，减小气泡占比
3. DeepSpeed


**混合精度问题**：<br>
混合精度训练时，很容易导致梯度爆炸，作者分析原因：
1. 由于混合精度训练，在attention层的score分布很容易超过FP16表示范围。作者针对attention做出的优化：<font color=#f00000>在attention的softmax计算时，先转换为FP32精度，计算完softmax后，再转换为FP16精度</font>。<br>
$softmax(\frac{Q_iK_i^T}{\sqrt{d}}) = softmax((\frac{Q_iK_i^T}{\alpha \sqrt{d}} - max(\frac{Q_iK_i^T}{\alpha \sqrt{d}})) \times \alpha)=\textcolor{red}{FP16}(softmax(\textcolor{red}{FP32}(\frac{Q_iK_i^T}{\alpha \sqrt{d}}) \times \alpha))$
2. 在训练初期，embedding层的梯度 与 其余层的梯度 有数量级的差异，例如：embedding层的梯度范围 $0.1~1$，其余层的梯度范围 $0.01 ~0.1$。作者的优化方法：<font color=#f00000>手动调小Embedding层的梯度</font><br>
$embedding = \alpha \times embedding  + (1 - \alpha) \times embedding.detach()$


**模型量化**：<br>
<p align="center"><img src="/datasets/posts/nlp/glm_int4.png" width=100% height=100%></p>

在保留中间结果为FP16的情况下，将GLM-130B模型进行量化，结果：
1. int8 下几乎没有任何损失，int4 只有极小的损失
2. Vector-wise 对称PTQ量化方案

## 相关

### 1、FlashAttention

Flash attention通过减少访问<font color=#f00000>HBM(high bandwidth memory)和on-chip SRAM内存读写时间</font>，提高计算速度的方法。具体来说: 1. 从HBM中加载输入数据；2. 在SRAM中执行所有的计算操作(矩阵乘法，mask，softmax，dropout，矩阵乘法)；3. 再将计算结果写回到HBM中。

1. <font color=#f00000>通过分块计算，增大每次计算矩阵的最小单元，从而降低对HBM的读写次数</font>，使得整体得到加速（HBM读写非常耗时）
    * 将数据从HBM按块load到SRAM
    * 按块在SRAM进行计算
    * 将结果更新会HBM

2. <font color=#f00000>通过重计算，降低内存</font>：被丢弃的变量在反传的过程中会再次使用到，需要重新计算得到，类似于梯度检查。
    * 在后向传播中不能存储中间注意力矩阵
    * 标准Attention算法的实现需要将计算过程中的S、P写入到HBM中，而这些中间矩阵的大小与输入的序列长度有关且为二次型，因此Flash Attention就提出了不使用中间注意力矩阵（指的是query*key, softmax(query*key)这些矩阵），通过存储归一化因子（用来辅助计算块注意力的参数）来减少HBM内存的消耗。
    * 虽然在反向传播时需要多次重新计算，增大了FLOPS，但是整体加速效果巨大。

3. 且通过对Q，K，V的分块，循环计算处理，拼接，最后得到的结果和标准attention一样，对用户相当于一个黑盒，这点其他加速方法很难做到。
4. FlashAttention的运行速度比PyTorch标准注意力快 2-4 倍，所需内存减少5-20倍。


### 2、Multi_query_attention
核心：所有 head 之间共享一份 key 和 value 的参数。原MHA中key 和 value 权重大小：$R^{k, d \times n}$，$n$ 表示head的个数；MQA中的key和value权重大小为 $R^{k, d}$ <br>
优势：能够在保证模型效果的同时加快 decoder 生成 token 的速度，速度相对于chatglm-1提升了一半。<br>

|计算量|Q|K|V|logits|O|Y|
|:--|:--|:--|:--|:--|:--|:--|
|MQA|bhndk|bmdk|bmdv|bhnmk|bhnmv|bhvnd|
|MHA|bhndk|bmhdk|bmhdv|bhnmk|bhnmv|bhvnd|

其中：b: batch-size；h：注意力头的数目；m，n：序列长度；k，v：每个注意力头的维度；d：模型隐藏层的维度。

