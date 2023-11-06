---
title: "LLaMa"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: LLaMa
    identifier: aigc-text-llama
    parent: aigc-text
    weight: 15
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","llama"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

<a href="https://zhuanlan.zhihu.com/p/648030318" target="bland">LLaMa2 翻译</a>

## 一、简介

**数据方面**<br>
1. LLaMa2训练了2000B的tokens，训练语料比LLaMa多了40%
    * 2000B 个token的预训练集，提供了良好的性能和成本权衡；对最真实的来源进行上采样，以增加知识并抑制幻觉，保持真实
    * 调查数据，以便用户更好地了解模型的潜在能力和局限性，保证安全。
2. 上下文长度从2048提升到了4096
3. LLaMa2-chat 模型还接受了超过100w的人类标注的训练数据
    * 开源数据选了 LLaMa2
    * 使用监督微调 LLaMa2-chat
    * 使用人类反馈强化学习(RLHF)进行迭代细化；包括拒绝采样、近端策略优化


**网络方面**<br>

LLaMa2 vs LLaMa，主要改动体现在 GQA 和 FFN 上:
1. 由MHA改成GQA：整体参数量会减少
2. FFN模块矩阵维度有扩充：增强泛化能力，整体参数量增加。


{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/nlp/llama_1.png" width=80% height=80%></p>
---
<p align="center"><img src="/datasets/posts/nlp/llama2_net.png" width=120% height=120%></p>
{{< /split >}}


1. RMSNorm 归一化
2. FFN中用swiGLU激活函数替换原来的Relu
3. 旋转位置编码 RoPE
4. 增加上下文长度
5. 分组查询注意力 GQA
    * 原始的 多头注意力：MHA
    * 具有单个KV投影的原始多查询格式：MQA
    * 具有8个KV投影的分组查询注意力变体：GQA

<p align="center"><img src="/datasets/posts/nlp/GQA_attention.png" width=90% height=90%></p>

**训练方面** <br>

1. 预训练细节：
    * 用AdamW优化器进行训练，其中： $β_1 =0.9，β_2 = 0.95，eps = 10−5$。
    * 使用余弦调整学习率，预热2000steps，$lr$ 衰减到峰值的10%
    * 使用0.1的权重衰减 、1.0的梯度裁剪

2. 精调细节：
    * 余弦学习率，$lr=2e-5$
    * 权重衰减0.1，batch_size=64，序列长度为4096
    * 训练2个epoch
    * 引入Ghost Attention 有助于控制多轮对话


## 二、网络结构
<p align="center"><img src="/datasets/posts/nlp/llama_scale.png" width="90%" height="90%"></p>

### 1、RoPE旋转位置编码
RoPE 不仅可以处理位置信息，还可以处理距离信息。因为旋转操作可以很好地反映出元素之间的相对位置关系。

### 2、RMSNorm

不用计算方差，速度提升了40%

### 3、SwiGLU

在原来Transformer的FFN中，是这样的：$FFN(x, W_1, W_2, b_1, b_2) = ReLU(xW_1 + b_1)*W_2 + b_2$ <br>
SwiGLU，是这样的: $SwiGLU(x, W, V, b, c) = Swish(xW + b) \otimes \sigma{(xV+c)}$。

SwiGLU其实就是采用Swish作为激活函数的GLU变体，GLU其实不是一个激活函数，而是一个网络层。<br>
由于多引入了一个线性表示，每个Layers 会增加 $4H^2$ 的参数量，LLaMa 是怎么客服这一点的呢？<br>
LLaMa把SwiGLU中的 $W, V, W_2$ 的矩阵维度从 $(dim, dim)$ 变成 $(dim, \frac{2}{3}dim)$，从而保证整体的参数量不变。代码如下：

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x))) * self.w3(x))
```
<p align="center"><img src="/datasets/posts/nlp/llama_p.png" width="90%" height="90%"></p>


### 4、GQA
<p align="center"><img src="/datasets/posts/nlp/llama_2.png" width="90%" height="90%"></p>

<p align="center"><img src="/datasets/posts/nlp/GQA_attention.png" width=90% height=90%></p>

MHA、GQA、MQA的区别与联系：
  * 原始的 多头注意力：MHA
  * 具有单个KV投影的原始多查询格式：MQA，多头之间可以共享 $K, V$，速度上可以减少 $30\\% - 40\\%$ 的吞吐。
  * 具有8个KV投影的分组查询注意力变体：GQA



## 三、训练

训练流程：
1. 先经过自监督训练，得到LLaMa2基座模型
2. 在有标签的数据上监督训练SFT
3. 再进行RLHF，其中使用拒绝采样和PPO算法
    * 在RLHF中，使用标注数据训练得到两个RW，对RLHF的模型评判，训练。


### 1、RW模型
**怎么标注数据**：<br>
1. 让标注人员写一个prompt，模型对这个prompt会有很多输出结果（为了保障答案的多样性，会用不同模型变体、调节不同的温度超参，来生成多样的结果），从输出结果中采样两条。
2. 标注人员对采样的两条结果做判别标注。同时，还要写出程度：明显更好、更好、稍好，忽略不计/不确定。
3. 搜集数据时，关注：有用性和安全性。例如，"提供制作炸弹的详细说明 "可能被认为是有用的，但根据我们的安全指南，这是不安全的。


其中，安全数据集的数据分布
1. 首选回答是安全的，而另一个回答不是， 占比18%
2. 两个回答都是安全的，占比47% 
3. 两个回答都是不安全，占比35%

**训练RW模型**：<br>
训练了两个独立的奖励模型（RW），一个针对帮助性helpfulness进行了优化，另一个针对安全性safety进行了优化。<br>

RW模型的平均准确率：65% ~ 70%，当标注人员的偏好共识很强时，准确率可达 80% ~ 90%<br>

用标注的 prompt + 标注结果，作为输入，输出一个score值来表示生成的质量。<br>

为什么要训练两个RM模型? <br>
（Bai 等人，2022a）等人发现，有用性和安全性，有时会相互抵消。


**RM模型的loss**:
$$
L = -log(\sigma[r_{\theta}(x, y_c) - r_{\theta}(x, y_r) - m(r)])
$$
其中 $r_{\theta}(x, y)$ : 表示模型的评分；$y_c$: 表示接受；$y_r$：表示拒绝；$m(r)$: 一个离散函数，表示两个答案的距离程度。

<p align="center"><img src="/datasets/posts/nlp/llama_4.png" width="90%" height="90%"></p>



### 2、RLHF

使用两种主要算法探索RLHF的微调：
1. 近端策略优化（Proximal Policy Optimization, PPO）
2. 拒绝采样 <br>
对模型的 $K$ 个输出进行采样，然后根据奖励选出最佳候选者。将选定的输出用于梯度更新。<br>
对于每个prompt，获的最高奖励分数的样本视为新的 Gold-standard。会在新的排序样本集上对模型进行微调，强化奖励。

**区别**:
1. 广度
    * 拒绝采样，对给定的prom，进行 $K$ 次采样；
    * 而PPO只进行一次采样
2. 深度
    * PPO，在第t步的训练过程中，样本是上一步梯度更新后第 $t-1$ 步更新模型策略的函数
    * 拒绝采样：对模型初始策略下的所有输出进行采样，收集新的数据集，然后再应用类似与SFT的微调。

### 3、GAtt方式训练
在对话设置中，有些指令适用于所有的对话回合，例如：简明扼要地回答、扮演某个任务。在最初的RLHF模型，往往会在几轮对话后忘记了最初的指令。<br>

为了解决这些局限性，提出了 ”幽灵注意力“ GAtt，做法也很简单：
1. 定义一个整个对话过程中都应遵守的指令 inst
2. 把这条指令合成到对话的所有用户信息中。
