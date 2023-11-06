---
title: "ELECTRA综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: ELECTRA综述
    identifier: electra-summary-github
    parent: electra-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["ELECTRA"]
categories: ["Basic"]
---


## 一、背景



## 二、ELECTRA
<a href="https://openreview.net/pdf?id=r1xMH1BtvB" target="blank">ELECTRA</a>的全称是Efficiently Learning an Encoder that Classifies Token Replacements Accurately。最主要的贡献是提出了新的预训练任务和框架，把生成式的Masked language model(MLM)预训练任务改成了判别式的Replaced token detection(RTD)任务，判断当前token是否被语言模型提换过。<br>

之前的方法，都需要预测一些部分。或者是预测下一个单词，或者是预测被盖住的部分。其实预测的模型需要的训练量是很大的，ELECTRA不做预测，只回答是或者否。<br>
比如: 上面原来的句子是“the chef cooked the meal”，现在把“cooked”换成了“ate”。ELECTRA需要判断输入的单词中，哪些被替换了。<br>
这样的好处是：预测Y/N简单；并且每个输出都被用到，可以计算损失。不像训练BERT时，只要mask的部分才计算loss。<br>
<p align="center"><img src="/datasets/posts/nlp/ELECTRA.png" width=70% height=70%></p>
ELECTRA的效果还比较不错，从上图可以看到，在同样的运算量下，它的表现比其他模型要好，并且能更快地达到较好的效果。<br>


**结构**
<p align="center"><img src="/datasets/posts/nlp/electra_2.png" width=90% height=90%></p>

1. 类似GAN的思路，生成器：随机mask，把mask位置的token随机替换成其他的token。

**优势**：
1. 训练速度比bert快，充分训练后，准确率更高。比如：效果与RoBerta一致，计算量只用了1/4。
2. 计算loss的时候，不但使用了mask部分，也使用了非mask的部分。




## 三、总结


