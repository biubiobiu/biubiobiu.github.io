---
title: "模型应用策略"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 模型应用策略
    identifier: aigc-summary-application
    parent: aigc-summary
    weight: 5
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","大模型", "应用策略"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介

要想训练一个针对特定领域的大模型，如果采用<font color=#f00000>全量参数微调（Full Parameter Futuing）</font>的方法，一方面需要大量的高质量数据集、另一方需要较高的算力，那么，有没有不需要大量算力就能在特定领域数据上对大模型进行微调的方法呢？<br>
下面，给大家介绍几种常见的大模型微调方法：
1. Adapter-Tuning
2. Prefix-Tuning
3. Prompt-Tuning(P-Tuning)、P-Tuning v2
4. LoRA


{{< alert type="success" >}}

对于大语言模型应用的两种不同的使用方式：
1. “专才”：只精通指定任务。怎么让一个基础模型在指定任务上比较精通呢？有两种方式：
    1. 加外挂：比如：在bert后面添加几个fc层，完成指定任务
    2. fine-tune：
    3. Adapter插件：固定原来模型，添加一个额外的模型插件。例如：Bitfit、AdapterBias、Houlsby、Prefix-tuning；ControlNet， LoRA，Text Inversion

2. “全才”：模型有各种背景知识，用户可以通过使用prompt指令，来要求模型按照指令输出。
    1. In-context Learning
    2. Instruction tuning
    3. Chain-of-Thought Prompting
    4. APE


{{< /alert >}}

### 1、Adapter插件

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/adaptor_0.png" width=100% height=100%></p>
---
github: <a href="https://github.com/google-research/adapter-bert" target="bland">adapter-bert</a> <br>
有人提出 <a href="https://arxiv.org/pdf/1902.00751.pdf" target="bland">Adaptor</a> 的概念，在预训练的模型中加入一些叫Apt(Adaptor)的层，在微调的时候，只微调Apt层。这篇文章中，将Adapter插在Feed-forward层之后，在预训练的时候是没有Adapter的，只有在微调的时候才插进去。并且在微调的时候，只调整Adapter层的参数。
{{< /split >}}


### 2、Prefix-tuning

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/prefix-tuning.png" width=100% height=100%></p>
---
github: <a href="https://github.com/XiangLi1999/PrefixTuning" target="bland">PrefixTuning</a> <br>
根据<a href="https://arxiv.org/pdf/2101.00190.pdf" target="bland">《Prefix-Tuning》</a> ，前缀调整实现了与微调所有层相当的建模性能，同时只需要训练 0.1% 的参数——实验基于 GPT-2 模型。此外，在许多情况下，前缀调整甚至优于所有层的微调，这可能是因为涉及的参数较少，这有助于减少较小​​目标数据集上的过度拟合。

{{< /split >}}

### 3、Prompt-tuning

github: <a href="https://github.com/THUDM/P-tuning" target="bland">P-tuning</a> <br>
论文：<a href="https://arxiv.org/pdf/2103.10385.pdf" target="bland">《GPT Understands》</a> <br>

<p align="center"><img src="/datasets/posts/nlp/prompt-1.png" width=100% height=100%></p>

> 1. 先将一些为prompt输入到LSTM中，用LSTM输出的向量来替换原始的Prompt token
> 2. 然后一起输入到 预训练模型中
> 3. LSTM和预训练模型一起训练


github: <a href="https://github.com/THUDM/P-tuning-v2" target="bland">P-Tuning v2</a> <br>
论文：<a href="https://arxiv.org/pdf/2110.07602.pdf" target="bland">《P-Tuning v2》</a> <br>

<p align="center"><img src="/datasets/posts/nlp/prompt-2.png" width=100% height=100%></p>

> p-tuning的改进版：不同层中的提示作为前缀token加入到输入序列中，并独立于其他层间(而不是由之前的transformer层计算)。<br>
> 1. 一方面，通过这种方式，P-tuning v2有更多的可优化的特定任务参数(从0.01%到0.1%-3%)，以允许更多的每个任务容量，而它仍然比完整的预训练语言模型小得多。
> 2. 另一方面，添加到更深层的提示可以对输出预测产生更直接和重大的影响，而中间的transformer层则更少

### 4、LoRA
{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/lora-0.png" width=100% height=100%></p>
---
github: <a href="https://github.com/microsoft/LoRA" target="bland">LoRA</a> <br>
根据<a href="https://arxiv.org/pdf/2106.09685.pdf" target="bland">《LoRA》</a> <br>

特点：
1. 训练速度更快
2. 计算量更低
3. 训练权重更小

{{< /split >}}

### 5、随便选一些参数

**实验**：随便选一些参数作为需要更新的，其他的参数冻结。<br>
**结果**：发现这样的结果与 全参数训练的结果也差不多。<br>

**说明**：这个可能是因为 模型太大了。模型增大到一定程度，模型结构就不是那么重要了。


## 二、大模型-使用范式

### 1、训练时

**Instruction Tuning 与 Prompt Tuning 的区别**

> Prompt Tuning <br>
> 针对每个任务，单独生成prompt模版，然后在每个任务上进行full-shot微调与评估，其中预训练模型参数是<font color=#f00000>冻结的</font>。<br>
> 在T5中：冻结预训练模型，只更新添加在第一层的soft prompt。在full-shot上就能和finetune上效果相当。<br>
> prompt tuning 的一系列方法和adapter越来越像了。

---

> Instruction Tuning <br>
> 针对每个任务，单独生成 Instruction（hard token），通过在若干个full-shot任务上进行微调，然后再具体的任务上进行评估泛化能力（zero-shot）。其中预训练模型参数是<font color=#f00000>不冻结的。</font> <br>
> 这两个方法的核心点：去挖掘语言模型本身具备的知识。不同点是：<br>
> 1. prompt 是激发语言模型的不全能力，比如：给出上半句生成下半句、完形填空
> 2. instruction 是激发语言模型的理解能力，通过给出更明显的指令，让模型去理解并做出正确的反馈。

#### 1. Prompt-tuning

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/prompt-0.png" width="100%" height="100%"></p>
---
设计prompt：<br>
每个任务都会设计一种prompt。

{{< /split >}}



##### 1. 人工设计prompt
根据具体任务，人工设计prompt。。。。

##### 2. 自动生成prompt
**APE** <br>
<a href="https://arxiv.org/pdf/2211.01910.pdf" target="bland">《LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS》</a> <br>

作者用大模型自己输出有用的prompt，通过筛选后，选出效果较好的。

<p align="center"><img src="/datasets/posts/nlp/cot_prompt_2.png" width="90%" height="90%"></p>

整体效果如下：
<p align="center"><img src="/datasets/posts/nlp/ape_prompt.png" width="90%" height="90%"></p>




#### 2. Instruction-tuning

##### 1. FLAN
来自Google公司Jaon Wei的<a href="https://arxiv.org/pdf/2109.01652.pdf" target="bland">《FINETUNED LANGUAGE MODELS ARE ZERO-SHOT
LEARNERS》</a> 2021-09-03 <br>

论文的结论：<font color=#f00000>模型可以学习到人类设定好的指令，并根据已学到的知识，在测试时对于未见过的指令，结果有好的表现</font>。<br>

**Motivation**：通过提升语言模型对Instructions的理解能力，来提高语言模型零样本学习能力。<br>

##### 2. T0
<a href="https://arxiv.org/pdf/2110.08207.pdf" target="bland">《Multitask Prompted Training Enables Zero-Shot Task Generalization》</a> 2021-10-15 <br>

> T0和 FLAN 工作整体相似，区别是：
> 1. 增加了任务（171个NLP任务） 和 prompt 数量（1939个prompt）
> 2. FLAN使用了decoder-only，T0使用了encoder+decoder
> 3. FLAN每次针对测试一个任务训练一个模型，其他任务作为训练集，T0为了测试模型泛化能力，只在多任务数据集上训练一个模型。证明了隐式多任务学习能提升模型泛化和zero-shot能力。

##### 3. RLHF

<a href="https://arxiv.org/pdf/2203.02155.pdf" target="bland">《Multitask Prompted Training Enables Zero-Shot Task Generalization》</a> 2022-03-04 <br>

**Motivation**：使用 人类反馈的强化学习（RLHF）技术，根据用户和API的交互结果，对模型的多个输出进行排序，然后再利用这些数据微调GPT-3，使得InstructGPT模型遵循指令方面比GPT-3更好。


#### 3. Delta-tuning

**Motivation**：只微调少量的参数，效果可以达到全参数微调差不多的效果。<br>
**可以这样理解**：因为有了比较好的预训练模型，后续就不许再重点学习知识点了。后续的下游任务，主要是如何激活跟该任务相关的知识点。<br>
**实现**：固定预训练模型不变，添加、修改等一些网络，训练时只更新这些网络的参数。

<p align="center"><img src="/datasets/posts/nlp/delta-tuning-0.png" width="90%" height="90%"></p>

构建Delta的方式：
1. 增量式（addition-based）：例如：Adapter、LoRA、prefix-Tuning
2. 指定式（Specification-base）：例如：BitFit（只微调bias）、随便选一些参数更新
3. 重参数化 （Reparameterization-base）：例如：降维、降秩


### 2、推理时

#### 1、In-Context Learning

##### 1. 解释1
<a href="https://arxiv.org/abs/2202.12837" target="bland">《Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?》</a> <br>
In-context learning是一种学习范式，它允许语言模型通过以演示形式组织的若干个示例或者指令来学习任务。In-context learning（ICL）的核心在于<font color=#f00000>从任务相关的类比样本中学习</font>，ICL要求若干示例以特定形式进行演示，然后将当前输入x跟上述示例通过prompt拼接到一起作为语言模型的输入。本质上，它利用训练有素的语言模型根据演示的示例来估计候选答案的可能性。简单理解，<font color=#f00000>就是通过若干个完整的示例，让语言模型更好地理解当前的任务，从而做出更加准确的预测</font>。
<p align="center"><img src="/datasets/posts/nlp/in_context_2.jpg" width="90%" height="90%"></p>

实验结论：
1. <font color=#f00000>ICL 中Ground Truth信息无关紧要</font>。<br>
作者实验对比：没有示例、多个示例-且label是一一对应的、多个示例-且label是随机的。对比发现：
    1. 随机label 与 正确label 的效果相当，性能只下降了 $ 0 - 5\\%$。
    2. 没有示例，效果下降较多。
    
<p align="center"><img src="/datasets/posts/nlp/in_context_1.jpg" width="90%" height="90%"></p>
2. ICL的性能收益主要来自 <font color=#f00000>独立规范的输入空间和标签空间，以及正确一致的演示格式</font>。<br>
作者实验了这4个因素：输入空间、标签空间、演示格式。对比实验：把输入换成外部语料；把标签换成英语单词；缺少输入或者label。实验发现：
    1. 把输入换成外部语料；把标签换成英语单词；缺少输入或者label。这些操作都会使得效果明显下降。
<p align="center"><img src="/datasets/posts/nlp/in_context_0.jpg" width="90%" height="90%"></p>

**个人理解**：比如做情感分析，示例：输入<sep>label。这种格式很重要，label是否正确不重要。<br>
大模型通过预训练，对文本时有理解能力的。ICL 的prompt中有多个 样例的格式，是让大语言模型知道当前是在做情感分析任务，而不是在做其他任务，按照情感分析的思路输出。并不是根据 prompt中的几个样例学习，而是唤醒机器要执行什么样的任务。


##### 2. 解释2
来自Google公司Jaon Wei的<a href="https://arxiv.org/pdf/2303.03846.pdf" target="bland">《LARGER LANGUAGE MODELS DO IN-CONTEXT
LEARNING DIFFERENTLY》</a> <br>
Google的这篇文章解释：大语言模型在ICL中是有学习的，解释1的结论之所以成立，是因为解释1用的模型还不够大，在更大的模型中，在ICL中的学习会表现的更为明显。



#### 2、Chain-of-Thought Prompting

来自Google公司Jaon Wei的<a href="https://arxiv.org/pdf/2201.11903.pdf" target="bland">《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》</a> <br>
发现：在推理的任务中，只是给一些示例，大模型的效果不好，如果<font color=#f00000>能给到推理的思路过程</font>，大模型的效果就会有明显提升。<br>
<p align="center"><img src="/datasets/posts/nlp/cot_prompt_0.png" width="90%" height="90%"></p>

CoT的变形：<br>
来自Google公司的<a href="https://arxiv.org/pdf/2205.11916.pdf" target="bland">《Large Language Models are Zero-Shot Reasoners》</a>，<a href="https://arxiv.org/pdf/2205.10625.pdf" target="bland">《LEAST-TO-MOST PROMPTING ENABLES COMPLEX
REASONING IN LARGE LANGUAGE MODELS》</a>  <br>
发现：由于这个推理思路是人工写的，这些数据量比较少，而且操作比较麻烦。作者的操作：
1. 在生成答案之前，加了一个要求：<font color=#f00000>Let's think step by step.</font>
2. 生成多个答案，然后投票

<p align="center"><img src="/datasets/posts/nlp/cot_prompt_1.png" width="90%" height="90%"></p>




