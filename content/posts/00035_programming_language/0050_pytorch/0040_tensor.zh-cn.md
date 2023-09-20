---
title: "Tensor和变量"
date: 2022-04-08T06:00:20+06:00
menu:
  sidebar:
    name: Tensor和变量
    identifier: torch-tensor
    parent: torch
    weight: 40
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["torch","Tensor"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

> **Tensor**
> 1. 每个张量Tensor都有一个相应的torch.Storage，用来保存数据。<br>torch.Storage: 是一个单一数据类型的连续一维数组。每个Tensor都有一个对应的相同数据类型的存储：class torch.FloatStorage
> 2. 类tensor：提供了一个存储 多维的、横向视图，并定义了数值运算。
> 3. torch.Tensor.abs()：会在原地计算，并返回改变后的tensor<br>
   torch.Tensor.abd()：在一个新的tensor中计算结果

> **变量**
> 1. Variable 在torch.autograd.Variable中，Variable的结构图：<p align="center"><img src="/datasets/posts/language/torch_variable.png" width="50%" height="50%" title="cuda" alt="cuda"></p>
    <font color=#a020f0>data</font>：Variable的tensor数值 <br>
    <font color=#a020f0>grad_fn</font>：表示得到这个Variable的操作，<br>
    <font color=#a020f0>grad</font>：表示Variable的反向传播梯度 <br>
    示例1：x = Variable(torch.Tensor([1]), requires_grad=Ture) <br>
    其中：requires_grad=True ：这个参数表示是否对这个变量求梯度。<br>
    <font color=#a020f0>x.backward()</font>：自动求导。自动求导不需要再去明确地写明那个函数对那个函数求导，直接通过这行代码就可以对所有的需要梯度的变量进行求导。<br>
    <font color=#a020f0>x.grad</font>：存放的就是x的梯度值 <br>
    示例2：y.backward(torch.FloatTensor([1,0.1,0.01]))，表示得到的梯度分别乘以1,0.1,0.01
> 2. Variable和Tensor本质上没有区别，不过Variable会被放入一个计算图中，然后进行前向传播、反向传播、自动求导。
> 3. tensor与Variable之间的转换： tensor —to—> Variable：b=Variable(a)



## 一、Tensor信息

1. [x] <font color=#a020f0>torch.is_tensor(obj)</font>	判断是否为tensor
2. [x] <font color=#a020f0>torch.is_storage(obj)</font>	判断obj是一个pytorch storage对象
3. [x] <font color=#a020f0>torch.set_default_tensor_type()</font>
4. [x] <font color=#a020f0>torch.numel(Tensor)</font>	返回张量中元素的个数


## 二、创建Tensor

1. [x] <font color=#a020f0>torch.Tensor([[1,2],[3,4]])</font>。创建---返回指定数值的张量
2. [x] <font color=#a020f0>torch.randn(*sizes, out=None)</font>。创建---返回标准正态分布的随机数张量。标准正态分布，形状由sizes定义
3. [x] <font color=#a020f0>torch.randperm(n, out=None)</font>。创建---返回0~n-1之间的随机整数1维张量。返回一个从0~n-1的随机整数排列
4. [x] <font color=#a020f0>torch.rand(*sizes, out=None)</font>。创建---返回[0, 1)的均匀分布张量
5. [x] <font color=#a020f0>torch.arange(start, end, step=1, out=None)</font>。创建---返回一个1维张量。[start, end) 以step为步长的一组序列值
6. [x] <font color=#a020f0>torch.range(start, end, step=1, out=None)</font>。创建---返回一个1维张量。[start, end) 以step为步长的1维张量
7. [x] <font color=#a020f0>torch.zeros(*sizes, out=None)</font>。创建---返回一个全为0的张量。生成一个tensor, 数值为0，形状由sizes定义
8. [x] <font color=#a020f0>torch.eye(n, m=none，out=None)</font>。创建---返回一个2维单位矩阵张量。对角线为1，其他位置为0。m默认为n
9. [x] <font color=#a020f0>torch.from_numpy(ndarray)</font>。创建---返回张量。将numpy.ndarray转换为Tensor。Tensor与ndarray共享同一个内存空间
10. [x] <font color=#a020f0>torch.linspace(start, end, steps=100, out=None)</font>。创建---返回一个1维张量。[start, end]间生成steps个样本
11. [x] <font color=#a020f0>torch.logspace(start, end, steps=100, out=None)</font>。创建---返回一个1维张量。[10^start, 10^end]间生成steps个样本


## 三、随机Tensor

1. [x] <font color=#a020f0>torch.manual_seed(seed)</font>。随机---种子。seed(int or long)
2. [x] <font color=#a020f0>torch.initial_seed()</font>
3. [x] <font color=#a020f0>torch.get_rng_state()</font>
4. [x] <font color=#a020f0>torch.set_rng_state(new_state)</font>
5. [x] <font color=#a020f0>torch.default_generator()</font>
6. [x] <font color=#a020f0>torch.bernoulli(input, out=None)</font>。随机---伯努利分布。<br>
input: 输入张量包含用于抽取上述二元随机值得概率。所以，输入的值在[0,1]区间。output: 与输入维度一致
7. [x] <font color=#a020f0>torch.multinomial(input,num_samples,replacement=False,out=None)</font>。随机---从多项分布中抽取num_samples个。<br>
input: 包含概率值得张量。num_samples(int): 抽取的样本数。replacement(bool): 决定是否能重复抽取。
8. [x] <font color=#a020f0>torch.normal(means,std)</font>。随机---正态分布抽取随机数。<br>
means(Tensor): 均值。std(Tensor): 标准差。输出: 与means, std 维度一致


## 四、切片Tensor

1. [x] <font color=#a020f0>torch.cat(inputs, dimension=0)</font>。切片---连接。在给定维度上对输入的张量序列进行连接操作。<br>
inputs: sequence of Tensors 任意相同类型的Tensor序列。dimension: int 沿着此维度连接张量序列
2. [x] <font color=#a020f0>torch.chunk(tensor, chunks, dim=0)</font>。切片---分块。tensor：待分块的输入张量。<br>
chunks(int)：分块的个数。dim(int)：沿着此维度进行分块
3. [x] <font color=#a020f0>torch.gather(input,dim,index,out=None)</font>。切片---聚合。沿给定轴，将输入索引张量index指定位置的值进行聚合
4. [x] <font color=#a020f0>torch.index_select(input,dim,index,out)</font>。切片---在dim轴方向，按照index下标取切片。<br>
input:输入张量。dim: 索引的轴。index: 包含索引下标的一维张量
5. [x] <font color=#a020f0>torch.nonzero(input,out=None)</font>。<br>
input: 输入张量。返回非零的坐标，输出格式：每行是一个非零的坐标。非零坐标
6. [x] <font color=#a020f0>torch.split(tensor,split_size,dim=0)</font>。切片---分割。<br>
split_size(int): 单个分块的形状大小。dim(int): 沿此维度进行分割
7. [x] <font color=#a020f0>torch.squeeze(input,dim=None,out=None)</font>。切片---把维度值为1的去掉。维度(A-1-B-1-C) ---> 维度(A-B-C)
8. [x] <font color=#a020f0>torch.unsqueeze(input, dim=None)</font>。切片---添加值为1的维度
9. [x] <font color=#a020f0>torch.stack(squence, dim=0)</font>。切片---拼接。沿着一个新维度对输入张量序列进行连接
10. [x] <font color=#a020f0>torch.t(input, out=None)</font>。转置。input: 输入2维张量
11. [x] <font color=#a020f0>torch.transpose(input,dim0,dim1,out=None)</font>。转置
12. [x] <font color=#a020f0>torch.unbind(tensor,dim=0)</font>。移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片
13. [x] <font color=#a020f0>torch.outer()</font>。内积。outer(a, b) = a<sup>T</sup>b
14. [x] <font color=#a020f0>torch.polar()</font>
15. [x] <font color=#a020f0>torch.view()</font> 相当于torch.reshape,  torch.resize。-1: 代表自动调整这个维度上的元素个数，保证总数不变。
16. [x] <font color=#a020f0>torch.unsqueeze(input,dim,out=None)</font>。切片---添加维度1。对输入的指定位置插入维度1

## 五、序列化

1. [x] <font color=#a020f0>torch.save(obj, f, pickle_module=<>, pickle_protocol=2)</font>	保存模型变量到磁盘<br>
obj: 保存对象。<br>
f: 类文件对象或者一个保存文件名的字符串。<br>
pickle_module: 用于pickling元数据和对象的模块。<br>
pickle_protocol: 指定pickle protocal可以覆盖默认参数	保存一个对象到一个硬盘文件<br>
2. [x] <font color=#a020f0>torch.load(f, map_location=None, pickle_module=<>)</font> 从磁盘加载模型变量<br>
f: 类文件对象或者一个保存文件名的字符串。<br>
map_location: 一个函数或字典规定如果remap存储位置。<br>
pickle_module:	从磁盘文件中读取一个通过torch.save保存的对对象

## 六、并行化

1. [x] <font color=#a020f0>torch.get_num_threads()</font>
2. [x] <font color=#a020f0>torch.set_num_threads(int)</font>

