---
title: 数据结果
weight: 220
menu:
  notes:
    name: 数据结构
    identifier: notes-algorithm-struct
    parent: notes-algorithm
    weight: 10
math: true
---

{{< note title="数据结构" >}}  
常见的数据结构可分为「线性数据结构」与「非线性数据结构」，具体为：「数组」、「链表」、「栈」、「队列」、「树」、「图」、「散列表」、「堆」。

<p align="center"><img src="/datasets/note/data-struct.png" width="80%" height="80%" title="data-struct" alt="data-struct"></p>

{{< /note >}}


{{< note title="数组与链表" >}}  

> 数组： 在内存中是连续的一整块。
> 1. 随机访问，数组在内存中是连续的一整块，所以支持随机访问。
> 2. 增/删操作，费事。增加元素时，如果内存不够一整块，还得整体迁移

---
> 链表： 可以存储在内存的任何地方。
> 1. 顺序访问，由于存在任何地方，每个元素都存储了下一个元素的地址，所以只能从头开始逐个查询。
> 2. 增/删操作，不费事。只要修改一下 <font color=#a020f0>下一元素地址</font> 就行。

{{< /note >}}


{{< note title="栈" >}}  
<font color=#a020f0>递归</font>操作，就是使用的调用栈。即：把每个递归调用函数，都压入栈，完成一个弹出一个，直到空栈。

{{< /note >}}

<!--=====================队列=================-->
{{< note title="队列" >}} 
<font color=#a020f0>队列</font>(First In First Out)：先进先出的数据结构。<br>
> 1. 图的广度优先搜索，就是先把1级元素压入队列，然后在一个一个出队遍历时，把其邻居压入队列
> 2. 

{{< /note >}}

<!--=====================散列表=================-->
{{< note title="散列表" >}}  
<font color=#a020f0>散列函数</font>：将任何输入映射到数字。<br>

在pyhton中 散列表的实现为字典 dict()

散列表是一种非线性数据结构，通过利用 Hash 函数将指定的「键 key」映射至对应的「值 value」，以实现高效的元素查找。<br>
比如：通过输入学号，在名字库里找到对应的名字。
```python
# 输入：学号
# 小力: 10001
# 小特: 10002
# 小扣: 10003
# 名字库
names = [ "小力", "小特", "小扣" ]
# Hash函数的目的：把学号，映射为序号index，
# 这个序号index就是 名字库names的名字对应序号
```
<p align="center"><img src="/datasets/note/hash.png" width="100%" height="100%" title="hash" alt="hash"></p>

{{< /note >}}


{{< note title="堆" >}}  
堆是一种基于「完全二叉树」的数据结构，可使用数组实现。以堆为原理的排序算法称为「堆排序」，基于堆实现的数据结构为「优先队列」。堆分为「大顶堆」和「小顶堆」，大（小）顶堆：任意节点的值不大于（小于）其父节点的值。

完全二叉树定义： 设二叉树深度为 k，若二叉树除第 k 层外的其它各层（第 1 至 k−1 层）的节点达到最大个数，且处于第 k 层的节点都连续集中在最左边，则称此二叉树为完全二叉树。

<p align="center"><img src="/datasets/note/heap.png" width="30%" height="30%" title="heap" alt="heap"></p>

上图就是一个「小顶堆」，堆的操作：
1. 搜索：$O(1)$，就是访问 堆顶的元素。
2. 添加：就是要满足堆的定义：<font color=#a020f0>任意节点的值不大于（小于）其父节点的值。</font>
3. 删除：跟添加一样，就是要满足堆的定义：<font color=#a020f0>任意节点的值不大于（小于）其父节点的值。</font>

{{< /note >}}

<!--
{{< note title="堆" >}}  
{{< /note >}}
-->