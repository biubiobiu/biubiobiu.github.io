---
title: Gluon实例
weight: 220
menu:
  notes:
    name: Gluon实例
    identifier: notes-mxnet-gluon-summary
    parent: notes-mxnet-gluon
    weight: 10
---

{{< note title="实例-单层感知机">}}

![单层感知机](/datasets/posts/dp_summary/single_perceptron.jpg)

模型：o = w<sub>1</sub>*x<sub>1</sub> + w<sub>2</sub>*x<sub>2</sub> + b </p>

输出`o`作为线性回归的输出，输入层是2维特征；输入层不涉及计算，该神经网络只有输出层1层。</p>
`神经元`：输出层中负责计算o的单元。</p>
该神经元，依赖于输入层的全部特征，也就是说输出层中的神经元和输入层中各个输入完全连接，所以，这里的输出层又叫作`全连接层(fully connected layer)`或者`稠密层(dense layer)`

{{< /note >}}

{{< note title="生成数据集">}}
目标： o = 2*x<sub>1</sub> - 3.4*x<sub>2</sub> + 4.2   其中：
样本集：features: [w<sub>1</sub>, w<sub>2</sub>]，  labels: [真实值+噪声]

```python
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

```
{{< /note >}}


{{< note title="读取数据集 - 从零实现">}}


```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        # take函数根据索引返回对应元素
        yield features.take(j), labels.take(j)  

```
{{< /note >}}

{{< note title="读取数据集 - Gluon实现">}}

`DataLoader` 返回一个迭代器，一次返回batch_size个样本
```python
from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

```
{{< /note >}}

{{< note title="模型定义 - 从零实现">}}

手动定义模型参数，一定要开辟存储梯度的内存。
```python
# 将权重初始化为：均值为0、标准差为0.01的正太随机数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
# 开辟存储梯度的内存
w.attach_grad()
b.attach_grad()
# 定义线性回归的模型
def linreg(X, w, b): 
    return nd.dot(X, w) + b



```
{{< /note >}}

{{< note title="模型定义 - Gluon实现">}}
`Gluon`模块：提供了大量预定义的层。`nn`模块(neural networks)的缩写，所以里面定义了大量神经网络 的层。
`Sequential`：可以看作是一个 串联各个层的容器，在构建模型时，在该容器中一次添加层。当给定输入数据时，
容器中的每一层的输出作为下一层的输入。</p>
`init`模块：initializer的缩写。该模块提供了模型参数初始化的各种方法。

```python
from mxnet.gluon import nn
from mxnet import init
# 定义模型
net = nn.Sequential()
net.add(nn.Dense(1))
# 初始化模型参数
net.initialize(init.Normal(sigma=0.01))
```
{{< /note >}}

{{< note title="定义损失函数 - 从零实现">}}

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
loss = squared_loss
```
{{< /note >}}

{{< note title="定义损失函数 - Gluon实现">}}
`Gluon`中的`loss`模块：定义了各种损失函数。
```python
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # 平方损失又称L2范数损失
```
{{< /note >}}

{{< note title="定义优化算法 - 从零实现">}}

`param.grad`自动求梯度模块计算得来的梯度是一个批量样本的梯度和。在迭代模型参数时，需要除以批量大小来得到平均值。
```python
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

```
{{< /note >}}

{{< note title="定义优化算法 - Gluon实现">}}

`Gluon`模块中的`Trainer`类，用来迭代模型中的全部参数。这些参数可以通过`collect_params`函数获取。
```python
from mxnet.gluon import Trainer
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```
{{< /note >}}

{{< note title="训练模型 - 从零实现">}}


```python
lr = 0.03
num_epochs = 3
net = linreg
# 训练模型一共需要num_epochs个迭代周期
for epoch in range(num_epochs): 
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # x和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

```
{{< /note >}}

{{< note title="训练模型 - Gluon实现">}}

通过`Trainer`实例的`step`函数来迭代模型参数。由于loss是长度为batch_size的向量，在执行l.backward()时，
等价于执行l.sum().backward()。所以要用batch_size做平均。
```python
num_epochs = 3
# 训练模型一共需要num_epochs个迭代周期
for epoch in range(1, num_epochs + 1):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # x和y分别是小批量样本的特征和标签
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)  # l是有关小批量X和y的损失
        l.backward() # 等价于l.sum().backward()
        trainer.step(batch_size)  # 指定batch_size，从而对批量样本梯度求平均
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

```
{{< /note >}}