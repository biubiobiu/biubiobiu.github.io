---
title: 二叉树-遍历
weight: 220
menu:
  notes:
    name: 二叉树-遍历
    identifier: notes-algorithm-tree
    parent: notes-algorithm
    weight: 20
math: true
---

{{< note title="Depth First Search(DFS)遍历" >}}  

> **深度优先遍历**：<br>
> 1. 使用<font color=#f00000>递归</font>，代码比较简单
> 2. 如果不用递归，可以利用<font color=#f00000>栈</font>这种数据结构



```python
# -*- coding: utf-8 -*-
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class Tree_Method:

    def DFS(self, root):
        '''
        深度优先遍历，即先访问根节点，然后遍历左子树接着遍历右子树。
        主要利用栈的特点，先将右子树压栈，再将左子树压栈，这样左子树就位于栈顶，
        可以结点的左子树先与右子树被遍历。
        '''
 
        if root == None:
            return None
        stack = []
        '''用列表模仿入栈'''
        stack.append(root)          
        while stack:
            '''将栈顶元素出栈'''
            current_node = stack.pop()
            print(current_node.value, end=' ')
            '''判断该节点是否有右孩子，有就入栈'''
            if current_node.right:
                stack.append(current_node.right)
            '''判断该节点是否有左孩子，有就入栈'''
            if current_node.left:
                stack.append(current_node.left)

    def preOrder(self, root):
        '''先序遍历'''
        if root == None:
            return None
        print(root.value)
        self.preOrder(root.left)
        self.preOrder(root.right)
 
    # 先序打印二叉树（非递归）
    def preOrderTravese(node):
        stack = [node]
        while len(stack) > 0:
            print(node.val)
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
            node = stack.pop()
 
    def minOrder(self, root):
        '''中序遍历'''
        if root == None:
            return None
        self.minOrder(root.left)
        print(root.value)
        self.minOrder(root.right)
 
    # 中序打印二叉树（非递归）
    def inOrderTraverse(node):
        stack = []
        pos = node
        while pos or stack:
            # 当前节点不为null，先入栈，然后继续检测其左子节点
            if pos:
                stack.append(pos)
                pos = pos.left
            # 当前节点为null，表示上一个节点的左子节点为null，
            # 1. 打印上一节点，
            # 2. 然后检测上一节点的右子节点
            else:
                pos = stack.pop()
                print(pos.val)
                pos = pos.right
 
    def postOrder(self, root):
        '''后序遍历'''
        if root == None:
            return None
        self.postOrder(root.left)
        self.postOrder(root.right)
        print(root.value)
 
    # 后序打印二叉树（非递归）
    # 使用两个栈结构
    # 第一个栈进栈顺序：左节点->右节点->跟节点
    # 第一个栈弹出顺序： 跟节点->右节点->左节点(先序遍历栈弹出顺序：跟->左->右)
    # 第二个栈存储为第一个栈的每个弹出依次进栈
    # 最后第二个栈依次出栈
    def postOrderTraverse(node):
        stack = [node]
        stack2 = []
        while stack:
            node = stack.pop()
            stack2.append(node)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        while stack2:
            print(stack2.pop().val)
```

{{< /note >}}


{{< note title="Breadth First Search(BFS)遍历" >}}

> **广度优先遍历**：<br>
> 1. 一般借助 <font color=#f00000>队列</font>，每层入队，遍历完后再把下一层copy到队列中。


```python
# -*- coding: utf-8 -*-
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
 
class Tree_Method:
    
    def create_tree(self, arr):
        '''
        利用二叉树的三个组成部分：根节点-左子树-右子树；
        传入的arr是一个多维列表，每一维最大为3，
        每一维中的内容依次表示根节点-左子树-右子树。然后递归的进行构建
        '''
 
        length = len(arr)  #计算每一维的大小
        root = TreeNode(arr[0]) #获取每一维的根节点
        if length >= 2:         #判断是否有左子树
            root.left = self.create_tree(arr[1])
        if length >= 3:         #判断是否有右子树
            root.right = self.create_tree(arr[2])
        return root
 
    def BFS(self, root):
        '''
        广度优先遍历，即从上到下，从左到右遍历。
        主要利用队列先进先出的特性，入队的时候，是按根左右的顺序，那么只要按照这个顺序出队就可以了
        '''
 
        if root == None:
            return None
        queue = []
        '''用列表模仿入队'''
        queue.append(root)          
        while queue:
            '''将队首元素出栈'''
            current_node = queue.pop(0)
            print(current_node.value, end=' ')
            '''判断该节点是否有左孩子，有就入队'''
            if current_node.left:
                queue.append(current_node.left)
            '''判断该节点是否有右孩子，有就入队'''
            if current_node.right:
                queue.append(current_node.right)
```

{{< /note >}}

<!------------------------------------------------>

{{< note title="堆排序" >}}

利用堆排序，时间复杂度可以是 $O(n log_2 n)$ <br>
堆排序还可以应用到：比如：<font color=#f00000>从N个数中，取最大的k个值</font>。<br>
思路：构建一个k维的<font color=#f00000>小顶堆</font>，这样堆顶就是这个k个数的最小值。从N个数中逐一取数，
1. 如果该数小于堆顶值，则丢弃
2. 如果该数大于堆顶值，则该数大小目前在前k。用该数替换堆顶，然后维护堆。

```python
# encoding: utf-8

# 大顶堆
def big_heap(array, start, end):
    root = start
    # 左孩子的索引
    child = root * 2 + 1
    while child <= end:
        # 节点有右子节点，并且右子节点的值大于左子节点，则将child变为右子节点的索引
        if child + 1 <= end and array[child] < array[child + 1]:
            child += 1
        if array[root] < array[child]:
            # 交换节点与子节点中较大者的值
            array[root], array[child] = array[child], array[root]
            # 交换值后，如果存在孙节点，则将root设置为子节点，继续与孙节点进行比较
            root = child
            child = root * 2 + 1
        else:
            break

# 小顶堆
def little_heap(array, start, end):
    root = start
    # 左孩子的索引
    child = root * 2 + 1
    while child <= end:
        # 节点有右子节点，并且右子节点的值小于左子节点，则将child变为右子节点的索引
        if child + 1 <= end and array[child] > array[child + 1]:
            child += 1
        if array[root] > array[child]:
            # 交换节点与子节点中较小者的值
            array[root], array[child] = array[child], array[root]
            # 交换值后，如果存在孙节点，则将root设置为子节点，继续与孙节点进行比较
            root = child
            child = root * 2 + 1
        else:
            break


# 正序：使用大顶堆
def heap_sort(array):
    first = len(array) // 2 - 1
    # 1.构建大顶堆：从下到上，从右到左对每个非叶节点进行调整，循环构建成大顶堆
    for start in range(first, -1, -1):
        big_heap(array, start, len(array) - 1)

    # 2.排序
    for end in range(len(array) - 1, 0, -1):
        # 交换堆顶和堆尾的数据
        array[0], array[end] = array[end], array[0]
        # 重新调整完全二叉树，构造成大顶堆
        big_heap(array, 0, end - 1)

    return array


# 倒序：使用小顶堆
def heap_sort_reverse(array):
    first = len(array) // 2 - 1

    # 1.构建小顶堆：从下到上，从右到左对每个非叶节点进行调整，循环构建成大顶堆
    for start in range(first, -1, -1):
        little_heap(array, start, len(array) - 1)

    # 2.排序
    for end in range(len(array) - 1, 0, -1):
        # 交换堆顶和堆尾的数据
        array[0], array[end] = array[end], array[0]
        # 重新调整完全二叉树，构造成大顶堆
        little_heap(array, 0, end - 1)
    return array


def main():
    print('#===run a program with a main function===#')
    array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
    rst = heap_sort(array)
    print(rst)

```

{{< /note >}}
<!--
{{< note title="torch模块-样例" >}}
{{< /note >}}
-->
