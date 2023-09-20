---
title: "基本概念"
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 基本概念
    identifier: math-basic-conception
    parent: math-probability-theory
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["概率论","基本概念"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

# 一、基本概念

{{< alert type="info" >}}

**随机实验**：$E$ <br>
**样本空间**：记为 $S$。随机实验 $E$ 的所有可能结果组成的集合，称为随机实验 $E$ 的样本空间。<br>
**样本点**：样本空间的元素，即：随机实验 $E$ 的每个结果。<br>

**随机事件**：随机实验 $E$ 的样本空间$S$的子集，称为 $E$ 的随机事件。<br>
**基本事件**：由单个样本点组成的单点集，成为基本事件。<br>
**必然事件**：样本空间 $S$ 集合，成为必然事件。<br>
**不可能事件**：空集 $\varnothing$。 <br>

**概率**：随机实验 $E$，样本空间为 $S$。对于 $E$ 的每一件事 $A$，概率 记为 $P(A)$ <br>
**条件概率**：事件 $A$ 已经发生的条件下，事件 $B$ 发生的概率。<br>

**划分**：随机实验 $E$，样本空间为 $S$。$B_1, B_2, ..., B_n$ 为 $E$ 的一组事件，若
1. $B_iB_j= \varnothing , i \ne j, i,j=1,2,...,n$
2. $B_1 \cup B_2 \cup ... \cup B_n = S$

则，称 $B_1, B_2, ..., B_n$ 为样本空间 $S$ 的一个划分。<br>

**贝叶斯公式**：随机实验 $E$，样本空间为 $S$。$B_1, B_2, ..., B_n$ 为 $E$ 的一个划分，$A$ 为 $E$的一个事件。且 $P(B_i) > 0, P(A) > 0$。则
$$P(B_i|A)=\frac{P(A|B_i)P(B_i)}{ \sum_{j=1}^n P(A|B_j)P(B_j) } , i=1,2,...,n
$$



{{< /alert >}}

# 二、

