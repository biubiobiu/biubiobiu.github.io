---
title: "GPT综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: GPT综述
    identifier: gpt-summary-github
    parent: gpt-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["GPT"]
categories: ["Basic"]
---


## 一、简介
<p align="center"><img src="/datasets/posts/vlp/vlp_s.png" width="70%" height="70%" title="ViT" alt="ViT"></p>

<a href="https://zhuanlan.zhihu.com/p/590311003" target="blank">参考</a>

评估指标：
  1. 困惑度：困惑度（perplexity）的基本思想是：给测试集的句子赋予较高概率值的语言模型较好,当语言模型训练完之后，测试集中的句子都是正常的句子，那么训练好的模型就是在测试集上的概率越高越好，公式如下  $PP(W)=P(w_1w_2...w_N)^{\frac{-1}{N}}$ 。由公式可知，句子概率越大，语言模型越好，迷惑度越小。困惑度p可以理解为，如果每个时间步都根据语言模型计算的概率分布随机挑词，那么平均情况下，挑多少个词才能挑到正确的那个
  2. Prompt ranking accuracy：这个指标的定义和评价方法，来自《Hierarchical Neural Story Generation》。主要是关注引导语和生成的故事之间的相关性。具体做法是：在测试集中选择一对（p，g），p表示引导语，g表示生成的故事，在随机选取其他的引导语p1-p9，然后计算p和g的likelihood。条件一：（p，g）的相似性比（p1，g）的相似性大。 那么就取10000个测试集中的（p，g），满足条件一的部分占比，就称为Prompt ranking accuracy。
  3. 句子嵌入的相似度：计算引导语和生成的故事的句子嵌入（用GloVe取每个词的平均嵌入值）的余弦相似度。
  4. 评价连贯性：连贯性的评价方法，来自《Modeling local coherence: An entity-based approach》，主要思想是，在测试数据集中，对于一个故事s0，选择前面15个句子，打乱顺序，生成14个乱序的故事s1-s14。然后用语言模型计算s0-s14的可能性。对于s1-s14，如果可能性大于s0，就称为反例。 错误率定义为反例的占比。
  5. 评价单词的重复性和rareness


## 二、GPT
<a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf" target="blank">GPT(2018-06)</a> 

## 三、GPT-2
<a href="https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf" target="blank">GPT-2(2019-02)</a>

  1. GPT-2去掉了fine-tuning层：不再针对不同任务分别进行微调建模，而是不定义这个模型应该做什么任务，模型会自动识别出来需要做什么任务。这就好比一个人博览群书，你问他什么类型的问题，他都可以顺手拈来，GPT-2就是这样一个博览群书的模型。在Pretrain部分基本与GPT方法相同，在Fine-tune部分把第二阶段的Fine-tuning有监督训练具体NLP任务，换成了无监督训练具体任务，这样使得预训练和Fine-tuning的结构完全一致。当问题的输入和输出均为文字时，只需要用特定方法组织不同类型的有标注数据即可代入模型，如对于问答使用“问题+答案+文档”的组织形式，对于翻译使用“英文+法文”形式。用前文预测后文，而非使用标注数据调整模型参数。这样既使用了统一的结构做训练，又可适配不同类型的任务。虽然学习速度较慢，但也能达到相对不错的效果。
  2. 增加网络参数：GPT-2将Transformer堆叠的层数增加到48层，隐层的维度为1600，参数量更是达到了15亿。(Bert的参数量也才只有3亿)。base版-12层-117M，medium版-24层-345M，large版-36层-774M，xl版-48层-1558M。
  3. 调整transformer：将layer normalization放到每个sub-block之前，并在最后一个Self-attention后再增加一个layer normalization。




## 四、GPT-3
<a href="https://arxiv.org/pdf/2005.14165.pdf" target="blank">GPT-3(2020-05)</a>


## 五、chatGPT

chatGPT(2022-12)


---
BART(Bidirectional and Auto-Regressive Transformers，双向自回归转换器)


prompt


Google T5 (Text-to-Text Transfer Transformer)


---

Masked language model(MLM)
Replaced token detection(RTD)


## 参考
<a href="https://cloud.tencent.com/developer/article/1877406" target="blank">GPT-chatbot</a>

