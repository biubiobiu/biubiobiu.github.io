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

有人提出 <a href="https://arxiv.org/pdf/1902.00751.pdf" target="bland">Adaptor</a> 的概念，在预训练的模型中加入一些叫Apt(Adaptor)的层，在微调的时候，只微调Apt层。这篇文章中，将Adapter插在Feed-forward层之后，在预训练的时候是没有Adapter的，只有在微调的时候才插进去。并且在微调的时候，只调整Adapter层的参数。
<p align="center"><img src="/datasets/posts/nlp/adaptor_0.png" width=70% height=70%></p>



## 二、大模型-使用策略

### 1、In-Context Learning

#### 1. 解释1
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


#### 2. 解释2
来自Google公司Jaon Wei的<a href="https://arxiv.org/pdf/2303.03846.pdf" target="bland">《LARGER LANGUAGE MODELS DO IN-CONTEXT
LEARNING DIFFERENTLY》</a> <br>
Google的这篇文章解释：大语言模型在ICL中是有学习的，解释1的结论之所以成立，是因为解释1用的模型还不够大，在更大的模型中，在ICL中的学习会表现的更为明显。


### 2、Instruction-tuning

#### 1. FLAN
来自Google公司Jaon Wei的<a href="https://arxiv.org/pdf/2109.01652.pdf" target="bland">《FINETUNED LANGUAGE MODELS ARE ZERO-SHOT
LEARNERS》</a> <br>
论文的结论：模型可以学习到人类设定好的指令，并根据已学到的知识，在测试时对于未见过的指令，结果有好的表现。


### 3、Chain-of-Thought Prompting

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


### 4、APE

<a href="https://arxiv.org/pdf/2211.01910.pdf" target="bland">《LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS》</a> <br>

作者用大模型自己输出有用的prompt，通过筛选后，选出效果较好的。

<p align="center"><img src="/datasets/posts/nlp/cot_prompt_2.png" width="90%" height="90%"></p>

整体效果如下：
<p align="center"><img src="/datasets/posts/nlp/ape_prompt.png" width="90%" height="90%"></p>

