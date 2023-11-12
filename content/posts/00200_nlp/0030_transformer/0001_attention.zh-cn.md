---
title: "Attention"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: Attention
    identifier: transformer-attention-github
    parent: transformer-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["NLP", "attention"]
categories: ["Basic"]
---


## 一、Attention机制

如何有选择地引导注意力：<br>
**非自主性提示：** 基于环境中物体的突出性和易见性。比如 《辛德勒的名单》中的镜头：黑白镜头中的穿红衣服的小女孩。<br>
**自主性提示：** 选择受到 认知、意识的控制。<br>


{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/NG8ySY5hHvCODEL.png" width="80%" height="80%" title="attention" alt="attention"></p>
在不受自我意识控制的情况下，与环境差别最大的事物，就越显眼、易见。<br>
在受到自我意识控制的情况下，意识偏向那个，就选择那个<br>
---
<p align="center"><img src="/datasets/posts/nlp/wr6U4eubTy3fqaJ.png" width="100%" height="100%" title="attention" alt="attention"></p>
查询(query)：自主性提示，类似于自我意识。<br>
键(key)：非自主提示，类似于事物的突出性、易见性。<br>
值(value)：感官输入，类似于具体的事物-值。<br>

{{< /split >}}

---

attention机制可以认为是一个这样的函数：<br>

$$ f(\bold{q_j}) = \sum_{i=1}^m \alpha(\bold{q}_j, \bold{k}_i) \bold{v}_i$$
由$ \bold{V}$ 的各个向量的加权平均，组成一个新的向量 $f(q_j)$。其中，权重的计算是通过 query向量和每个key向量 计算出来的，这个计算方式可以有多种，比如：`加性注意力、缩放点积注意力`

$\bold{Q} \in \R^{n \times q}$: 查询矩阵，是由N个向量组成，每个向量有q个元素<br>
K-V: M个键值对集合。<br>
$\bold{K} \in \R^{m \times k}$: `M个键向量`组成的矩阵，每个键向量(k维)：就是每个字的标签信息<br>
$\bold{V} \in \R^{m \times v}$: `M个值向量`组成的矩阵，每个值向量(v维)：就是每个字的embeding<br>

### 1、加性注意力

$$\alpha(\bold{q}_j, \bold{k}_i) = \bold{w}_v^T tanh(\bold{W}_q \bold{q}_j + \bold{W}_k \bold{k}_i)$$
其中，$\bold{w}_v^T \in \R^h, \bold{W}_q \in \R^{h \times q}, \bold{W}_k \in \R^{h \times k}$ 是需要训练的。

### 2、缩放点积注意力(SDPA)
attention机制的SDPA(缩放点积注意力Scaled Dot-Product Attention)实现方式，计算效率更高，但是点积操作要求 $\bold{q}$ 和 $\bold{k}$ 具有相同的长度。<br>
假设：$\bold{q}$ 和 $\bold{k}$ 的所有元素都是独立的随机变量，并且满足标准正态分布 $N(0,1)$<br>
那么：两个向量的点积，服从正态分布 $N(0, d)$，其中 $d$ 就是$\bold{q}$(或者$\bold{k}$)的长度。<br>
所以：点积后，除以 $\sqrt{d}$，即：
$$\alpha(\bold{q}_j, \bold{k}_i) = \frac{\bold{q}_j^T \bold{k}_i}{\sqrt{d}}$$
基于n个查询、m个键值对，计算注意力，其中：
$\bold{Q} \in \R^{n \times d}$、$\bold{K} \in \R^{m \times d}$、$\bold{V} \in \R^{m \times v}$ 的缩放点积注意力：
$$softmax(\frac{\bold{Q} \bold{K}^T}{\sqrt{d}}) \bold{V} \in \R^{n \times v}$$

具体操作：<br>
1. $\bold{Q}$的每个向量 $\bold{q}_i$ 做如下操作:
    1. 计算第i个向量 $q_i$ 与M个键向量的相似度(内积)，生成一个1*M的向量
    2. 对该向量做softmax操作(概率化)
    3. 用概率化后的值做`M个值向量`权重系数，做加权求和，生成一个加权后的embeding
2. $\bold{Q}$的向量个数：表示需要多少个加权后的embeding，即：$\tilde{V}$

<p align="center"><img src="/datasets/posts/nlp/SDPA.jpg" width="60%" height="60%" title="SDPA" alt="SDPA"></p>

### 3、多头注意力(MHA)

在实践中，当给定 query、key、value时，我们希望模型可以基于相同的注意力机制，学习到不同的行为，然后将不同的行为作为知识组合起来，以捕获序列内各种范围的依赖关系。因此，允许注意力机制组合使用 query、key、value的`不同子空间表示`，可能是有益的。所以，对给定的query、key、value，经过不同的`线性变换`获取其子空间表示，然后并行地送入注意力机制，最后把各个子空间的输出拼接起来，再通过一个可以学习的线性变换产生最终输出。

MHA(多头注意力Multi-Head Attention) 实现方式：多路融合的SDPA，具体操作：<br>
1. 对Q、K、V矩阵做多次线性变换，例如：第i次变换的生成结果 $Q'_i, K'_i, V'_i$
2. 利用第i次线性变换后的 $Q'_i, K'_i, V'_i$，做SDPA操作，得到 $\tilde{V_i}$
3. 对所有的 $\tilde{V_i}$，在列方向上concat拼接起来

<p align="center"><img src="/datasets/posts/nlp/MHA.jpg" width="60%" height="60%" title="MHA" alt="MHA"></p>

### 4、实际模式
|QKV的关系||
|:---|:---|
|$Q \neq K \neq V$|QKV模式|
|$Q \neq K=V$|QVV模式|
|$Q=K=V$|VVV模式，即：`自注意力`，自己即是查询向量，也是key向量；表示句子内部与自己相似的权重比较大|


### 5、在seq2seq的应用

在seq2seq架构中，编码器生成各个时间步的上下文变量state，最后一时间步的state作为解码器的state。然而，有个问题：在解码器 解码某个词元时，并非所有输入词元都需要，或者说并非所有输入词元的贡献都一样，肯定是有的输入词元的贡献大一些。所以，在解码时能不能让贡献大的输入词元的state权重大一些呢？<br>

<a href="https://arxiv.org/abs/1409.0473" target="blank">Bahdanau</a>等人提出了一个没有严格单向对齐限制的可微注意力模型。在预测词元时，如果不是跟所有输入词元都相关，模型使用`仅跟当前预测相关的部分输入序列`。
<p align="center"><img src="/datasets/posts/nlp/djMJCF9H3ZLY41x.png" width="60%" height="60%" title="MHA" alt="MHA"></p>

$$c_{t'} = \sum_{t=1}^T \alpha(s_{t'-1}, h_t) h_t$$
在解码时，需要 上一时间步的隐状态 $s_{t'-1}$ 和 上一时间步的真实值。添加attention的话，就要修改 $s_{t'-1}$，让其是 编码器各个隐状态的加权和，这就是attention的操作，即：

  - 用解码器上一时间步的隐状态 $s_{t'-1}$ 作为查询
  - 编码器各个隐状态 $h_t$ 其中 $t \in [1, n]$
  - $\alpha()$ 函数，采用`加性注意力`



## 参考

<a href="https://cloud.tencent.com/developer/article/1868051?from=10680" target="blank">Transformer详解</a>


