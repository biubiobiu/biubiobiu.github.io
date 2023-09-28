---
title: "生成式-问题"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 生成式-问题
    identifier: aigc-summary-hallucination
    parent: aigc-summary
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","生成式", "幻觉"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---


## 一、简介

- [ ] 问题1：在文本生成是，模型会一本正经的胡说八道，这种现象叫做模型的幻觉。
- [ ] 问题2：训练一个大模型，需要多少数据量呢？
- [ ] 问题3：数据预处理，怎么过滤、去重
- [ ] 问题4：模型大小 与 数据大小 的关系？

## 二、模型问题

### 1、Calibration

- [x] 问题1：在文本生成是，模型会一本正经的胡说八道，这种现象叫做模型的幻觉。<br>
<a href="https://arxiv.org/pdf/2207.05221.pdf" target="bland">《Language Models (Mostly) Know What They Know》</a> <br>
这篇论文发现：模型够大后，说谎才会心虚。
  1. 对于大模型，模型输出是正确的概率 VS 模型的自信度，这两个是相关的。当模型比较自信时，输出的结果是正确的概率就比较大。
  2. 对于小模型，模型输出是正确的概率 VS 模型的自信度，这两个是不相关的
<p align="center"><img src="/datasets/posts/nlp/calibration_0.png" width="100%" height="100%"></p>
其中，横轴：模型输出时的自信程度；纵轴：模型输出是正确的概率。黄色表示最大模型，自身表示最小模型。

## 三、数据问题
- [x] 问题2：训练一个大模型，需要多少数据量呢？<br>
训练一个大模型，需要多少数据量呢？<a href="https://arxiv.org/pdf/2011.04946.pdf" target="bland">《When Do You Need Billions of Words of Pretraining Data?》</a> <br>
<p align="center"><img src="/datasets/posts/nlp/train_data_0.png" width="70%" height="70%"></p>


- [x] 问题3：数据预处理，怎么过滤、去重?<br>
数据预处理：<a href="https://arxiv.org/pdf/2112.11446.pdf" target="bland">《Scaling Language Models: Methods, Analysis & Insights from Training Gopher》</a> <br>
  1. 过滤有害的内容，通过Google的审核接口
  2. 去掉一些 HTML 前端的一些tag
  3. 规则过滤，去掉低质量的文本。
  4. 去重
  5. 剔除测试数据


- [x] 问题4：模型大小 与 数据大小 的关系？
<a href="https://arxiv.org/pdf/2203.15556.pdf" target="bland">《Training Compute-Optimal Large Language Models》</a> <br>
这篇文章发现：
    1. 小模型+大数据 和 大模型+小数据，这两个极端都不好。中间某个点事比较合适。这个合适的点：<font color=#f00000>大概是：模型参数：63B 匹配 数据量：1.4T</font>
<p align="center"><img src="/datasets/posts/nlp/model_data_size_1.png" width="90%" height="90%"></p>根据这个结论，作者训练了一版模型：Chinchilla。70B模型参数，1.4T数据量。跟以前的模型对比：Chinchilla在绝大多数的任务上，效果较好。
<p align="center"><img src="/datasets/posts/nlp/model_data_size_2.png" width="90%" height="90%"></p>

|model name|model size|tokens|
|:--|:--|:--|
|LaMDA(2022)|137B|168B|
|GPT-3(2020)|175B|300B|
|Jurassic(2021)|178B|300B|
|Gopher(2021)|280B|300B|
|MT-NLG(2022)|530B|270B|

