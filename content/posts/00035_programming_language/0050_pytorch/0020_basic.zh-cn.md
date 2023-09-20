---
title: "基础操作"
date: 2022-04-08T06:00:20+06:00
menu:
  sidebar:
    name: 基础操作
    identifier: torch-basic
    parent: torch
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["torch","基础操作"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数据类型

### 1、torch的数据类型

torch.Tensor 是默认的torch.FloatTensor 的简称。<br>
> 剥离出一个Tensor参与计算，不参与求导：Tensor后加 .detach()<br>
> 各个数据类型之间的转换：
> 1. 方法一：在Tensor后加，.long(), .int(), .float(), .double()
> 2. 方法二：可以用 .to()函数

|数据类型|CPU tensor|GPU tensor|
|:--|:--|:--|
|32-bit <font color=#a020f0>float</font>|torch.FloatTensor|torch.cuda.FloatTensor|
|64-big <font color=#a020f0>float</font>|torch.DoubleTensor|torch.cuda.DoubleTensor|
|16-bit <font color=#a020f0>float</font>|N/A|torch.cuda.HalfTensor|
|8-bit <font color=#a020f0>integer(unsigned)</font>|torch.ByteTensor|torch.cuda.ByteTensor|
|8-bit <font color=#a020f0>integer(signed)</font>|torch.CharTensor|torch.cuda.CharTensor|
|16-bit <font color=#a020f0>integer(signed)</font>|torch.ShortTensor|torch.cuda.ShortTensor|
|32-bit <font color=#a020f0>integer(signed)</font>|torch.IntTensor|torch.cuda.IntTensor|
|64-bit <font color=#a020f0>integer(signed)</font>|torch.LongTensor|torch.cuda.LongTensor|

### 2、Tensor与numpy

> 1. Tensor 转 Numpy: <font color=#a020f0>data.numpy()</font>
> 2. Numpy 转 Tensor：<font color=#a020f0>torch.from_numpy(data)</font>


### 3、Tensor与python数据

> 1. Tensor 转 单个python数据：<font color=#a020f0>data.item()</font>
> 2. Tensor 转 list ：<font color=#a020f0>data.to_list()</font>

### 4、Tensor数据位置

> 1. CPU张量 --转--> GPU张量：<font color=#a020f0>data.cuda()</font>
> 2. GPU张量 --转--> CPU张量：<font color=#a020f0>data.cpu()</font>

## 二、魔术命令%
|操作|解释|
|:--|:--|
|<font color=#a020f0>%timeit a.sum()</font>|检测某条语句的执行时间|
|<font color=#a020f0>%hist</font>|查看输入历史|
|<font color=#a020f0>%paste</font>|执行粘贴板中的代码|
|<font color=#a020f0>%cat ***.py</font>|查看某一个文件的内容|
|<font color=#a020f0>%run -i **.py</font>|执行文件，-i 代表在当前命名空间中执行|
|<font color=#a020f0>%quickref</font>|显示快速参考|
|<font color=#a020f0>%who</font>|显示当前命名空间中的变量|
|<font color=#a020f0>%debug</font>|进入调试模式</font>|q键退出|
|<font color=#a020f0>%magic</font>|查看所有魔术命令|
|<font color=#a020f0>%env</font>|查看系统环境变量|
|<font color=#a020f0>%xdel</font>|删除变量并删除其在Ipython上的一切引用|


## 三、

## 四、


