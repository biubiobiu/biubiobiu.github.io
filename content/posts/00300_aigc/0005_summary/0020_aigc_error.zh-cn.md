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
产生幻觉的原因：
1. LLM缺乏相关知识，或者内化了错误的知识
2. LLM有时高估了自己的能力
3. 问题对齐过程误导LLM进入幻觉：在对齐过程中接受针对它们在预训练阶段尚未获得的知识的指示时，实际上是一种不对齐过程，鼓励LLMs产生幻觉。
4. LLMs采用的生成策略存在潜在风险：LLMs有时会过分坚持早期的错误，即使它们意识到这是不正确的。换句话说，LLMs可能更喜欢为了自身一致性而堆积幻觉，而不是从错误中恢复。

减轻幻觉的方案：
1. 整理训练集：在预训练期间减轻幻觉主要集中在预训练语料库的策划上
2. SFT：监督训练，构建训练数据是减轻幻觉的一种方法
3. RLHF：人类监督强化学习。让模型学习到：诚实性、
4. 在推理阶段：
    * 设计解码策略
    * 利用外部知识来减轻LLMs中的幻觉

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


## 四、推理

- [x] 问题1：RoPE 外推？比如llama1的上下文长度为2048，对于2048之后的位置超出了训练2048的长度，模型推理时，该部分很可能就随机乱猜了，导致生成的结果不好。<br>
解决方案：<a href="https://zhuanlan.zhihu.com/p/646022309" target="bland">参考</a> ，本质上，将比例设置为原始模型上下文长度/当前序列长度。
1. 线性插值：将超出的部分通过线性插值压缩到2048。比如：f(x, m) = f(x, m/2)。这样只需要用少量4096长度的数据微调，就能达到很好的效果。<br>
该方法的缺陷是需要进行一定量的微调，让模型来适应这种改变。
2. NTK
3. 动态插值算法
4. NBCE：使用朴素贝叶斯扩展LLM的Context处理长度。苏神提出的，<a href="https://kexue.fm/archives/9617" target="bland">参考</a>。

- [x] 问题2：模型推理产生幻觉。<br>
解决方案：<a href="https://juejin.cn/post/7229891752647950394" target="bland">参考</a> <br>
1. 数据：构建边界数据集，比如：不知道
2. 通过推理后的score值
3. 人类反馈的强化学习
