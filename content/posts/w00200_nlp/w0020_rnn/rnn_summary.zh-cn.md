---
title: "RNN综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: RNN综述
    identifier: rnn-summary-github
    parent: rnn-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["循环神经网络"]
categories: ["Basic"]
---


## 一、文本预处理

### 1、词元-token
英文：在训练文本模型时，模型输入最小单元：可以是词元维度，也可以是字符维度(这样的话，模型还得学习怎么用字符组合成单词)<br>
中文：一般是字符维度；如果是词元维度，在模型之前需要进行`分词`，如果要使用词元维度，需要先分词，用空格间隔开。<br>
特殊词元：未知词元 `<unk>`，填充词元`<pad>`，序列开始词元 `<bos>`，序列结束词元 `<eos>`

### 2、词表-vocabulary
把token映射到：一个从0开始的数字索引，也就是：<br>
token --> idx：token_to_idx {0:then, 1:token, ....}<br>
idx --> token：idx_to_token: [the, token, ....] <br>

{{< alert type="info" >}}
例如：<br>
tokens:  例如：一篇文章<br>
例如：`[[一句话按照空格split后], [], [], ....]`

vocab：词表，代码里可以写成一个类，其元素有：<br>
self.idx_to_token ：[`'<unk>'`, 'the', ...]    token的列表，按照token的个数降序排列<br>
self.token_to_idx ：{`'<unk>'`: 0, 'the': 1, ....}   token-->idx 的映射<br>

corpus：语料库，先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为语料<br>
例如：`[('<unk>', 1000), ('the', 900), ....]`
{{< /alert >}}


## 二、深度循环神经网络

循环神经网络(Recurrent Netural Networks)：是具有隐状态的神经网络。<br>
类似于MLP多层感知机，RNNs只是添加了时间轴信息。比如，MLP的表示如下：<br>
$$ H = \phi(XW_{xh} + b_h) $$
$$O = HW_{hq} + b_q $$

RNNs的表示如下，需要$H_{t}$ 这个隐状态 记录历史：
$$ H_{t} = \phi(X_{t}W_{xh} + H_{t-1}W_{hh} + b_h)$$
$$ O_{t} = H_{t}W_{hq} + b_{q}$$

假设语料库的大小为N，那么RNNs的每次预测，其实就是一个N分类。所以，评估一个语言模型的好坏，用的是交叉熵：
$$\frac{1}{n}\sum_{t=1}^n-log P(x_t|x_{t-1},\dots,x_1)$$
由于历史原因，喜欢用`困惑度perplexity`来表示：
$$exp(\frac{1}{n}\sum_{t=1}^n-log P(x_t|x_{t-1},\dots,x_1))$$

<p align="center"><img src="https://s2.loli.net/2022/05/09/4jyDt6vrRQZflUI.png" width="50%" height="50%" title="hmm" alt="hmm"></p>

**RNNs的应用：** 文本生成、文本分类、问答/机器翻译、Tag生成；其输入/输出形式如下：

<p align="center"><img src="https://s2.loli.net/2022/05/14/hODV3ZxIwt2KXmR.jpg" width="50%" height="50%" title="hmm" alt="hmm"></p>

## 三、双向循环神经网络
**场景**：填空题，”下文“传达了重要信息，这些重要信息关乎到选择那些词来填空。<br>
我__ <br>
我__饿了 <br>
我__写作中 <br>

**设计方案**：概率图模型，设计一个隐马尔科夫模型：<br>
  1. 在任意时间步t，存在某个隐变量 $h_t$，通过概率 $P(x_t|h_t)$ 控制观测到的 $x_t$。
  2. 任何 $h_t \rarr h_{t+1}$ 转移，都是由一些状态转移概率 $P(h_{t+1}|h_t)$ 给出。

<p align="center"><img src="https://s2.loli.net/2022/05/09/73mVSQq5tFzaPJ9.png" width="50%" height="50%" title="hmm" alt="hmm"></p>

  3. 前向递归(forward recursion)：$\pi_{t+1} = f(\pi_t, x_t)$ 其中 $f$ 表示一些可被学习的函数。看起来就像循环神经网络中，隐变量的更新过程。这是前向计算
  4. 后向递归(backward recursion)：$\rho_{t-1} = g(\rho_t, x_t)$ 其中 $g$ 表示一些可被学习的函数。这是后向计算，知道未来数据何时可用，对隐马尔科夫模型是有益的。

<p align="center"><img src="https://s2.loli.net/2022/05/09/7YVUXcxF9sebqQL.png" width="50%" height="50%" title="hmm" alt="hmm"></p>

双向循环神经网络(bidirectional RNNs)：添加了反向传递信息的隐藏层。
  1. 在训练阶段，能够利用过去、未来的数据来估计现在空缺的词；在测试阶段，只有过去的数据，因此精度将会很差
  2. 双向循环神经网络的计算速度非常慢

所以双向循环神经网络并不常用。