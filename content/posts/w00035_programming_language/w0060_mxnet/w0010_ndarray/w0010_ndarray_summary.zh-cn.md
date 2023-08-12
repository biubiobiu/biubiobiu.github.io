---
title: "NdArray使用"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: NdArray使用
    identifier: posts-mxnet-ndarray-summary
    parent: posts-mxnet-ndarray
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["mxnet","NdArray"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、查阅文档

怎么查阅相关文档？ <a href="https://mxnet.apache.org/" target="blank">官网</a>  

### 1. 查阅`模块`里的所有函数和类
```python
from mxnet import nd
print(dir(nd.random))
```
1. __开头和结尾的函数 (python的特别对象) 可以忽略
2. _开头的函数 (一般为内部函数) 可以忽略
3. 其余成员，可以根据名字 大致猜出是什么意思。

### 2. 查阅特定`函数和类`的使用

想了解某个函数或者类的具体用法，可以使用help函数。以NDArray中的ones_like函数为例。
```python
help(nd.ones_like)
```

注意：</p>
1. jupyter记事本里，使用`?`来将文档显示在另外一个窗口中。例如：`nd.ones_like?` 与 `help(nd.ones_like)`效果一样。`nd.ones_like??`会额外显示该函数实现的代码。


## 二、内存开销

1. 原始操作 </p>
首先来个例子：Y = Y + X     -->  每个操作会新开内存来存储运算结果。
上例中，X，Y 变量首先存储在内存中，相加的计算结果会另外开辟内存来存储；然后变量Y在指向新的内存。
内存使用情况：</p>
内存id_x <-- X </p>
内存id_y <-- Y </p>
内存id_x+y <-- Y

2. Y[:] = X + Y 或者 Y += X </p>
通过`[:]`把X+Y的结果写进Y对应的内存中。上述操作中，需要另外开辟内存来存储计算结果。
内存使用情况：</P>
内存id_x <-- X </p>
内存id_y <-- Y </p>
内存id_x+y  --> 把`内存id_x+y`中数值复制到`内存id_y`中

3. 使用运算符全名函数中的out参数 </p>
可以避免临时内存开销，使用运算符全名函数：`nd.elemwise_add(X, Y, out=Y)`。内存使用情况： </p>
内存id_x <-- X </p>
内存id_y <-- Y </p>
内存id_y <-- 直接存放 X+Y 的计算结果

## 三、自动求梯度

MXNet提供的autograd模块，可以自动求梯度(gradient) </p>
```python
from mxnet import autograd, nd
# 1. 创建变量 x，并赋初值
x = nd.arrange(4).reshape((4, 1))
# 2. 为了求变量x的梯度，先调用attach_grad函数来申请存储梯度所需要的内存 
x.attach_grad()
# 3. 为了减少计算和内存开销，默认条件下MXNet是不会记录：求梯度的计算，
#    需要调用record函数来要求MXNet记录与求梯度有关的计算。
print(autograd.is_training())    # False
with autograd.record():
  print(autograd.is_training())  # True
  y = 2*nd.dot(x.T, x)
# 4. 调用backward函数自动求梯度。y必须是一个标量，
#  如果y不是标量：MXNet会先对y中元素求和，然后对该和值求有关x的梯度
y.backward() 

```
注意：</p>
1. 在调用record函数后，MXNet会记录并计算梯度；
2. 默认情况下，autograd会改变运行模式：从预测模式转为训练模式。可以通过调用is_training函数来查看。

