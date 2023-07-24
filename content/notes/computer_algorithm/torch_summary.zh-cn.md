---
title: 二叉树-遍历
weight: 220
menu:
  notes:
    name: 二叉树-遍历
    identifier: notes-algorithm-tree
    parent: notes-algorithm
    weight: 20
---

{{< note title="Depth First Search(DFS)遍历" >}}  

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
            if pos:
                stack.append(pos)
                pos = pos.left
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

{{< note title="torch模块-样例" >}}

```python
import torch
```
{{< /note >}}
