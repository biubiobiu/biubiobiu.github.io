---
title: "GRU网络"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: GRU网络
    identifier: rnn-gru-github
    parent: rnn-github
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["循环神经网络", "GRU"]
categories: ["Basic"]
---


## 一、简介

RNNs中，需要的信息都放在隐藏层，当序列太长时，隐藏层累积了太多的信息，对前面太久的信息，就不容易获取到了。<br>
另外，有些信息不太重要，有些词比较重要，所以，设计了：<br>
**更新门：** $Z_t$ 有助于捕获序列中的长期依赖关系。当$Z_t = 0$时，并不是就没有$H_{t-1}$的信息了，而是$H_{t-1}$的信息通过正常的计算$H_t$的途径进来；而当$Z_t > 0$时，$H_{t-1}$的信息可以绕过正常的计算途径，直接添加到$H_t$中。<br>

**重置门：** $R_t$ 有助于捕获序列中的短期依赖关系。$\tilde{H_t}$ 的计算跟RNNs计算相似，就是加了 $R_t$ 来限制 $H_{t-1}$，本来RNNs对太久的信息就不容易获取，所以 $R_t$ 的作用：是否忘掉历史没用的信息。<br>


$$R_t = sigmoid(X_tW_{xr}+H_{t-1}W_{hr}+b_r)$$
$$Z_t = sigmoid(X_tW_{xz}+H_{t-1}W_{hz}+b_z)$$
$$\tilde{H_t} = tanh(X_tW_{xh} + (R_t \odot H_{t-1})W_{hh} + b_h)$$
$$H_t = Z_t \odot H_{t-1} + (1-Z_t)\odot \tilde{H_t}$$

<p align="center"><img src="/datasets/posts/nlp/JvhC21NaOfyGjFp.jpg" width="50%" height="50%" title="hmm" alt="hmm"></p>

其中，$R_t$ ：表示`在更新候选隐状态时，需要多少历史隐状态信息`，$Z_t$ ：表示`在算真正的隐状态时，需要多少新输入的`$X_t$`的信息`，这两个的维度与隐状态是一致的。
