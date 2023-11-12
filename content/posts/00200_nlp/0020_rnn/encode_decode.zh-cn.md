---
title: "编解码架构"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 编解码架构
    identifier: rnn-edncode-github
    parent: rnn-github
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["循环神经网络", "编码器-解码器 架构"]
categories: ["Basic"]
---

## 一、编码器-解码器 架构

机器翻译：是把一个序列转换为另一个序列。为处理这种类型的输入和输出，设计这样的架构：
  - 编码器：接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。<br>
  - 解码器：将固定形状的编码状态映射到长度可变的序列。<br>

<p align="center"><img src="/datasets/posts/nlp/3PseiSrBRD4dJfX.jpg" width="50%" height="50%" title="architecture" alt="architecture"></p>


## 二、seq2seq

<a href="https://arxiv.org/abs/1409.3215" target="blank">Ilya Sutskever</a> 等人设计的seq2seq：将编码器最后一时间步的state，作为解码器第一时间步的state使用。<br>
<a href="https://arxiv.org/abs/1406.1078" target="blank">Kyunghyun Cho</a> 等人设计的seq2seq，将编码器最后一时间步的state，作为解码器每一个时间步的输入序列的一部分。

<p align="center"><img src="/datasets/posts/nlp/3lj9UytSvAMnhEI.jpg" width="70%" height="70%" title="seq2seq" alt="seq2seq"></p>


