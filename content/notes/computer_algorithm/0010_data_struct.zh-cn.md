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
<font color=#a020f0>递归</font>操作，就是使用的调用栈。即：把每个递归调用函数，都压入栈，完成一个弹出一个，直到空栈。<br>

> 优先考虑用栈的 “信号”
> 1. 有返回上一步的操作
> 2. 成对匹配的问题，比如：（）
> 3. 链表/list 的<font color=#f00000>翻转问题</font>，例如：<a href="https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2&envId=top-interview-150" target="blank">K 个一组翻转链表
</a> 

{{< /note >}}

<!--=====================队列=================-->
{{< note title="队列" >}} 
<font color=#a020f0>队列</font>(First In First Out)：先进先出的数据结构。<br>
> 1. 图的广度优先搜索，就是先把1级元素压入队列，然后在一个一个出队遍历时，把其邻居压入队列
> 2. 

{{< /note >}}


<!--=====================树=================-->
{{< note title="树" >}}

**字典树**<br>
字典树 ( Trie 树 ) 又称单词查找树，
是一种用于在 字符串集合 中 高效地 存储 和 查找 字符串 的 树形 数据结构。
<p align="center"><img src="/datasets/note/trie_tree.png" width="100%" height="100%" title="hash" alt="hash"></p>

实例： <a href="https://leetcode.cn/problems/implement-trie-prefix-tree/?envType=study-plan-v2&envId=top-interview-150" target="blank">实现 Trie (前缀树)</a>
```python
class Trie(object):

    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

    def searchPrefix(self, prefix):
        node = self
        for ch in prefix:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node


    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True


    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        node = self.searchPrefix(word)
        return node is not None and node.is_end


    def startsWith(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        node = self.searchPrefix(prefix)
        return node is not None
```


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

**小顶堆**： $K_i <= K_{2i+1} \ \\& \  K_i <= K_{2i+2}$ <br>
**大顶堆**： $K_i >= K_{2i+1} \ \\& \  K_i >= K_{2i+2}$ <br>

怎么构建一个「小顶堆」?<br>
1. 遍历二叉树的<font color=#f00000>非叶子节点自下往上</font>的构造小顶堆，针对每个非叶子节点，都跟它的左右子节点比较，把最大的值换到这个子树的父节点。

上图就是一个「小顶堆」，堆的操作：
1. 搜索：$O(1)$，就是访问 堆顶的元素。
2. 添加：就是要满足堆的定义：<font color=#a020f0>任意节点的值不大于（小于）其父节点的值。</font>
3. 删除：跟添加一样，就是要满足堆的定义：<font color=#a020f0>任意节点的值不大于（小于）其父节点的值。</font>

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

```


{{< /note >}}


<!-------------------graph------------------>
{{< note title="图" >}}

**连通图与非联通图**

<p align="center"><img src="/datasets/note/graph_1.png" width="100%" height="100%" title="graph" alt="graph"></p>

1. 非连通图：比如图a中，找不到一条从a到c的路径
2. 连通图：图b是一个连通图，因为从一个顶点到另一个顶点都至少存在一条通路

**生成树**

<p align="center"><img src="/datasets/note/graph_2.png" width="100%" height="100%" title="graph" alt="graph"></p>

所谓生成树，是指具备一下条件的<font color=#f00000>连通图</font>

1. 包含图中所有顶点
2. 任意顶点之间<font color=#f00000>有且只有一条通路</font>，比如上图就是一个连通图，对应的多种生成树。

**最小生成树**：就是上述生成树中，路径权值和最小的那个。具体问题比如：修路问题，n座城市之间修公路，要求两两互联，公里数最短。<br>
求解最小生成树的算法通常有两种：
1. <a href="http://c.biancheng.net/algorithm/prim.html" target="blank">普里姆算法<a>
2. <a href="http://c.biancheng.net/algorithm/kruskal.html" target="blank">克鲁斯卡尔算法<a>



{{< /note >}}


<!--
{{< note title="堆" >}}
{{< /note >}}
-->