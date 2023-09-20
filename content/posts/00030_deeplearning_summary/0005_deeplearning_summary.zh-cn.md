---
title: "深度学习开篇"
date: 2021-08-05T12:30:40+08:00
description: Markdown rendering samples
menu:
  sidebar:
    name: 深度学习开篇
    identifier: deep-learning-start
    parent: deep-learning-summary
    weight: 5
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["机器学习","深度学习","简介"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---


<a href="https://openaccess.thecvf.com/menu" target="blank">论文入口</a>

## 一、机器学习
{{< alert type="info" >}}
目前，人工智能研究领域主要体现在一下几个方面：
1. 智能感知：通过模拟人的感知能力（视觉、听觉、嗅觉）对外部信息进行感知和识别，并能够对信息进行加工和处理，从而做出反应。
2. 智能学习：学习是人工智能的主要标志和获取知识的重要手段，研究机器通过模拟人的学习能力，如何从小样本、大数据中学习，主要有：
    * 监督学习：（Supervised Learning）表示机器学习的数据是带有标记的，这些标记可以包括：数据类别、数据属性、特征点位置等。这些标记作为预期效果，不断修正机器的预测结果。常见的监督学习有<font color=#f00000>分类、回归、结构化学习</font>。
    * 半监督学习：（Semi-Supervised Learning）利用少量标注数据和大量无标注数据进行学习的方式。常用的半监督学习算法有：<font color=#f00000>自训练、协同训练</font>
    * 非监督学习：（Unsupervised Learning）表示机器学习的数据是没有标记的。常见的无监督学习有：<font color=#f00000>聚类、降维</font>
    * 强化学习：（Reinforcement Learning）通过智能体和环境的交互，不断学习并调整策略的机器学习算法。这种算法带有一种激励机制，如果智能体根据环境做出一个正确的动作，则施予一定的“正激励”；如果是错误的动作，则给与一定的“负激励”。通过不断地累加激励，以获取激励最大化的回报。做火热的应用就是 <font color=#f00000>AlphaGo Zero</font>

3. 认知推理：模拟人的认知能力，主要研究知识表示、推理、规划、决策等，主要有自然语言处理、脑科学。

{{< /alert >}}

## 二、表征学习
{{< alert type="success" >}}
**表征**：为了提高机器学习系统的准确率，需要将输入信息转化为有效的特征，或者更一般性地称为 <font color=#f00000>表征（Representation）</font> <br>

**表征学习**：如果有一种算法可以自动地学习有效的特征，并提高最终机器学习模型的性能，那么这种学习就可以叫做 <font color=#f00000>表征学习</font>。
表征学习的关键是解决语义鸿沟（Semantic Gap）问题。即：输入数据的底层特征与高层语义信息之间的不一致性和差异性。<br>
机器学习中经常用两种方式表示特征：局部表示(Local Representation)、分布式表示(Distributed Representation)。<br>
比如：颜色的表示。
1. 局部表示：也称为离散表示或者符号表示，比如：one-hot向量的形式。假设所有颜色 构成一个词表 $\bold V$，此时，可以用一个 $|\bold V|$ 维的one-hot向量来表示一中颜色。但是，one-hot向量的维数很高，且不能扩展，如果有一种新的颜色，就需要增加一维来表示。不同颜色之间的相似度都为0，无法直到“红色”和“中国红”的相似度要高于“红色”和“黑色”的相似度。
2. 分布式表示：另一种表示颜色的方法是用RGB值来表示颜色，不同颜色对应RGB三维空间中的一个点。分布式表示通常可以表示<font color=#f00000>低维的稠密向量</font>。

**嵌入**：神经网络将高维的<font color=#a020f0>局部表示空间</font> $\R^{|\bold V|}$，映射到一个非常低维的<font color=#a00000>分布式表示空间</font> $\R^{D}$。在这个低维空间中，每个特征不再是坐标轴上的点，而是分散在整个低维空间中，在机器学习中，这个过程也成为<font color=#a00000>嵌入（Embedding）</font>。比如：自然语言中词的分布式表示也经常叫做词嵌入。<br>

要学习到一种好的高层次语义表示（一般为分布式表示），通常只有从底层特征开始，经过多步非线性转换才能得到。<font color=#a00000>深层结构</font>的优点是可以提高特征的重用性，从而指数级增强表示能力。因此，<a href="https://arxiv.org/pdf/1206.5538.pdf" target="bland">表示学习的关键是构建具有一定深度的多层次特征表示</a>。



{{< /alert >}}

## 三、深度学习

深度学习是机器学习的一个重要的、新的研究领域，源于对神经网络的进一步研究，通常采用包含多个隐藏层的神经网络结构，目的是建立、模拟人脑学习过程。<br>

在描述深度学习之前，先回顾下机器学习和深度学习的关系。

1. **机器学习**：研究如何使用计算机系统利用经验改善性能。在机器学习的众多研究方向中，`表征学习`关注如何自动找出表示数据的合适方式，以便更好地将输入变换为正确的输出。

2. **深度学习**：是具有多级表示的表征学习方法。在每一级，深度学习通过简单的函数将该级的`表示`变换为更高级的`表示`。因此，深度学习模型也可以看作是由许多简单函数复合而成的函数。当这些复合函数足够多时，就可以表达非常复杂的变换。<br>
作为表征学习的一种，深度学习将自动找出每一级表示数据的合适方式。逐级表示越来越抽象的概念或模式。<font color=#a00000>高层特征是由底层特征通过推演归纳得到。</font><br>
深度学习可通过学习一种深层非线性网络结构来表征输入数据，实现复杂函数逼近，具有很强的从少数样本集中学习数据集本质特征的能力。深度学习的主要思想：通过自学习的方法，学习到训练数据的结构，并在该结构上进行有监督训练微调。
以图像为例，它的输入是一堆原始像素值，模型中逐级表示为：
{{< mermaid align="left" >}}
graph LR;
    A(特定位置和角度的边缘) --> B(由边缘组合得出的花纹)
    B --> C(由多种花纹进一步汇合得到的特定部位)
    C --> D(由特定部位组合得到的整个目标)
{{< /mermaid >}}

### 1、神经元

神经元模型：
1. 每个神经元都是一个多输入、单输出的信息处理单元
2. 神经元输入分兴奋性输入和抑制性输入两种类型
3. 神经元具有空间整合特性和阈值特性
4. 神经元输入与输出间有固定的时滞，主要取决于突触延迟
5. 忽略时间整合作用和不应期
6. 神经元本身是非时变的，即：其突触时延和突触强度均为常数

{{< mermaid align="center" >}}
graph LR;
    A1(x<sub>1</sub>) --> |输入| B1(W<sub>k1</sub>)
    A2(x<sub>2</sub>) --> |输入| B2(W<sub>k2</sub>)
    A3(x<sub>3</sub>) --> |输入| B3(W<sub>k3</sub>)
    B1 --> |权值| C(求和节点)
    B2 --> |权值| C(求和节点)
    B3 --> |权值| C(求和节点)
    C(求和节点) --> |v<sub>k</sub>| D(激活函数)
    D --> |y<sub>k</sub>| E(输出)
{{< /mermaid >}}

## 四、学习方式

### 1、多阶段

在一些复杂任务重，传统机器学习方法需要将一个任务的输入和输出人为地切割成很多子模块（或者多个阶段），每个子模块分开学习。比如：要完成一个自然语言理解任务，一般需要：
{{< mermaid align="left" >}}
graph LR;
    A(分词) --> B(词性标注)
    B --> C(句法分析)
    C --> D(语义分析)
    D --> E(语义推理)
{{< /mermaid >}}

这种学习方式有两个问题：
1. 每个模块都需要单独优化，并且其优化目标和任务总体目标并不能保证一致。
2. 错误传播，即：前一步的错误会对后续的模型造成很大的影响。

### 2、端到端

训练过程中不进行分模块或分阶段训练，而是直接优化任务的总体目标。中间过程不需要人为干预，无需其他额外信息。因此，端到端学习，需要解决贡献度分配问题。目前大部分采用神经网路模型的深度学习都是端到端学习。


## 五、学术会议

|简称|介绍|
|:--|:--|
|ICLR|国际表征学习大会（International Conference on Learning Representations）: 主要聚焦深度学习|
|NeurIPS|神经信息处理系统大会（Annual Conference on Neural Information Processing Systems）：交叉学科会议，但偏重于机器学习，主要包括神经信息处理、统计方法、学习理论及应用|
|ICML|国际机器学习会议（International Conference on Machine Learning）：机器学习顶级会议。深度学习作为近年来的热点，也占据了ICML|
|IJCAI|国际人工智能联合会议（International Joint Conference on Artificial Intelligence）：人工智能领域顶尖的综合性会议，历史悠久，从1969年开始举办。|
|AAAI|国际人工智能协会（AAAI Conference on Artificial Intelligence）：人工智能领域的顶级会议，每年二月份左右召开，一般在北美。|
||人工智能的子领域 - 专业学术会议|
|CVPR|IEEE国际计算机视觉与模式识别会议（IEEE Conference on Computer Vision and Pattern Recognition）|
|ICCV|计算机视觉国际大会（International Conference on Computer Vision）|
|ACL|国际计算语音学协会（Annual Meeting of the Association for Computational Linguistics）|
|EMNLP|自然语言处理实证方法会议（Conference on Empirical Methods in Natural Language Processing）|
