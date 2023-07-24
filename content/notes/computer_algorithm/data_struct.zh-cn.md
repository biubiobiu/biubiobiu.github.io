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


{{< note title="散列表" >}}  
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
2. 添加：就是要满足堆的定义：`任意节点的值不大于（小于）其父节点的值。`
3. 删除：跟添加一样，就是要满足堆的定义：`任意节点的值不大于（小于）其父节点的值。`

{{< /note >}}