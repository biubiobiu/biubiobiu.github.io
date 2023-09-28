---
title: "GPT综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: GPT综述
    identifier: gpt-summary-github
    parent: gpt-github
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["GPT"]
categories: ["Basic"]
---


## 模型评估

评估指标：
  1. 困惑度：困惑度（perplexity）的基本思想是：给测试集的句子赋予较高概率值的语言模型较好,当语言模型训练完之后，测试集中的句子都是正常的句子，那么训练好的模型就是在测试集上的概率越高越好，公式如下  $PP(W)=P(w_1w_2...w_N)^{\frac{-1}{N}}$ 。由公式可知，句子概率越大，语言模型越好，迷惑度越小。困惑度p可以理解为，如果每个时间步都根据语言模型计算的概率分布随机挑词，那么平均情况下，挑多少个词才能挑到正确的那个
  2. Prompt ranking accuracy：这个指标的定义和评价方法，来自《Hierarchical Neural Story Generation》。主要是关注引导语和生成的故事之间的相关性。具体做法是：在测试集中选择一对（p，g），p表示引导语，g表示生成的故事，在随机选取其他的引导语p1-p9，然后计算p和g的likelihood。条件一：（p，g）的相似性比（p1，g）的相似性大。 那么就取10000个测试集中的（p，g），满足条件一的部分占比，就称为Prompt ranking accuracy。
  3. 句子嵌入的相似度：计算引导语和生成的故事的句子嵌入（用GloVe取每个词的平均嵌入值）的余弦相似度。
  4. 评价连贯性：连贯性的评价方法，来自《Modeling local coherence: An entity-based approach》，主要思想是，在测试数据集中，对于一个故事s0，选择前面15个句子，打乱顺序，生成14个乱序的故事s1-s14。然后用语言模型计算s0-s14的可能性。对于s1-s14，如果可能性大于s0，就称为反例。 错误率定义为反例的占比。
  5. 评价单词的重复性和rareness


## 一、简介

基于文本预训练的GPT-1，GPT-2，GPT-3三代模型都是采用的以Transformer为核心结构的模型，不同的是模型的层数和词向量长度等超参，它们具体的内容如下：
|模型|发布时间|层数|head|hidden|参数量|预训练数据量|
|:--|:--|:--|:--|:--|:--|:--|
|GPT-1|2018年6月|12|12|768|1.17亿|5GB|
|GPT-2|2019年2月|48|-|1600|15亿|40GB|
|GPT-3|2020年5月|96|96|12888|175B|45TB|



## 二、GPT
<a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf" target="blank">GPT(2018-06)</a> <br>

其创造性的提出以Transformer的解码器来训练生成式模型，后面Bert的作者估计是看到了这篇论文，据说两个月时间就发表了以Transformer编码器训练的Bert模型。总结下GPT-1模型：

1. GPT-1 使用了一个仅有解码器的 Transformer 结构，每一个作为一个Layer，共有12层；
2. 使用了一个 768 维的嵌入向量来表示输入序列中的每个词或标记，使用了 12 个并行的注意力头（attention heads）；
3. 使用Adam优化器进行模型训练，在训练过程中，使用了学习率的 warmup 阶段和余弦退火调度机制，以平衡训练速度和模型性能；
4. 模型权重被初始化为均值为 0、标准差为 0.02 的正态分布（N(0, 0.02)），使用字节对编码（Byte Pair Encoding，BPE）来对文本进行分词处理，分词后得到的词汇表大小为 40000；
5. 激活函数是 GELU；
6. 文本输入序列固定长度是512；
7. 参数量 117M;
8. 使用了学习得到的位置嵌入向量(position embedding)，而不是Attention is All You Need中使用的正弦位置嵌入向量；

## 三、GPT-2
<a href="https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf" target="blank">GPT-2(2019-02)</a>

**GPT-2的改进**:

GPT-2 是GPT语言模型开始变大的地方，这是 OpenAI 第一次训练超过 1B 个参数的模型。
<font color=#f00000>通过提升模型的规模，来凸显GPT的优势</font>。在 GPT-1 中，作者训练了单个模型，但在这里，作者训练了一系列模型。
与GPT-1相比，架构上有如下差异：
<p align="center"><img src="/datasets/posts/nlp/gpt1-2.png" width=70% height=70%></p>

1. 层归一化操作，有原来的post-norm换成了pre-norm，<font color=#f00000>以加速训练和提高模型性能</font>。此外，在最后一个自注意力块的输出上添加了额外的层归一化；
2. 在权重初始化时，通过 $\frac{1}{\sqrt n}$ 进行缩放。这种缩放<font color=#f00000>有助于减少梯度更新的方差，使训练过程更加稳定</font>；
3. 扩大了其词汇表的大小，词汇表大小约为 <font color=#f00000>50,000</font>（相比于约 40,000）；
4. 增大文本输入序列长度 1024（相比于 512）这使得模型能够更好地理解和生成更长的文本；
5. batch size大小为 512（相比于 64）较大的批次大小有助于提高训练效率和模型并行计算的能力。
6. 最大的模型具有约 15 亿个参数。
7. 数据集：GPT-2 构造了一个新数据集，<a href="https://paperswithcode.com/dataset/webtext" target="bland">WebText</a>。全部来自于 Reddit 的外链，而且是那些获得至少三个赞的外链，经过清洗、去重后，得到8百万网页共计 <font color=#f00000>40GB 文本数据</font>。
  WebText 数据集的特点在于<font color=#f00000>全面而干净</font>。
  


**GPT-2的不同版本**:
|模型|Layers|d_size|ff_size|Heads|Parameters|
|:--|:--|:--|:--|:--|:--|
|GPT2-base|12|768|3072|12|1.17亿|
|GPT2-medium|24|1024|4096|16|3.45亿|
|GPT2-large|36|1280|5120|20|7.74亿|
|GPT2-xl|48|1600|6400|25|15.58亿|


## 四、GPT-3
<a href="https://arxiv.org/pdf/2005.14165.pdf" target="blank">GPT-3(2020-05)</a> <br>
GPT-3是大语言模型开始受到关注的开始。在论文中，作者训练了 10 个模型，参数范围从 1.25亿 个参数（“GPT-3 Small”）到 175B 个参数（“GPT-3”）。
<p align="center"><img src="/datasets/posts/nlp/gpt-3-model.png" width=90% height=90%></p>

在GPT-3中，模型的架构与GPT-2完全相同。唯一的区别是它们在transformer的各层中使用了<font color=#f00000>“交替的稠密和本地带状稀疏注意力模式”</font>。简单来说，GPT-3在注意力机制上进行了优化，引入了稀疏注意力的概念。<br>

> 1. 传统的点积注意力在计算复杂度上较高，而稀疏注意力可以提供更高的扩展性，并且在处理长序列时具有更高的效率。这种改进使得GPT-3能够更好地处理长文本输入，并且在计算效率和模型表现方面有所提升。
>2. GPT-3引入稀疏注意力的原因尚不清楚，也许是因为计算限制造成的，论文中并没详细的说明如何如何使用模型并行性训练模型，使得论文更难以复现。

## 五、chatGPT

chatGPT(2022-12)<br>
<a href="https://arxiv.org/pdf/2203.02155.pdf" target="bland">InstructGPT</a> <br>

ChatGPT的博客中讲到ChatGPT和InstructGPT的训练方式相同，不同点仅仅是它们采集数据上有所不同，但是并没有更多的资料来讲数据采集上有哪些细节上的不同。

## 六、GPT-4

<a href="https://arxiv.org/abs/2303.08774" target="bland">《GPT-4 Technical Report》</a>



---
BART(Bidirectional and Auto-Regressive Transformers，双向自回归转换器)


prompt


Google T5 (Text-to-Text Transfer Transformer)


---

Masked language model(MLM)
Replaced token detection(RTD)


## 参考
<a href="https://cloud.tencent.com/developer/article/1877406" target="blank">GPT-chatbot</a>


