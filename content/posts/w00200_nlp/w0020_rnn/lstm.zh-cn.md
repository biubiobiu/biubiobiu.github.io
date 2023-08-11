---
title: "LSTM网络"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: LSTM网络
    identifier: rnn-lstm-github
    parent: rnn-github
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["循环神经网络", "LSTM"]
categories: ["Basic"]
---


## 一、简介
长短期记忆网络(LSTM)

忘记门：$F_t = sigmoid(X_tW_{xf}+H_{t-1}W_{hf}+b_f)$ <br>
输入门：$I_t = sigmoid(X_tW_{xi}+H_{t-1}W_{hi}+b_i)$ <br>
输出门：$O_t = sigmoid(X_tW_{xo}+H_{t-1}W_{ho}+b_o)$ <br>
候选记忆单元：$\tilde{C_t} = tanh(X_tW_{xc} + (R_t \odot H_{t-1})W_{hc} + b_c)$ <br>
记忆单元：$C_t = F_t \odot C_{t-1} + I_t\odot \tilde{C_t}$ <br>
隐状态：$H_t = O_t \odot tanh(C_t)$ <br>

其中，$F_t, I_t, O_t, C_t, H_t, \in \R^{n \times d}$

<p align="center"><img src="https://s2.loli.net/2022/05/14/1kXM3OJbFujNL5o.jpg" width="50%" height="50%" title="hmm" alt="hmm"></p>


