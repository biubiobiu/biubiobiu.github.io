---
title: 常见算法思路
weight: 220
menu:
  notes:
    name: 常见算法思路
    identifier: notes-algorithm-recursion
    parent: notes-algorithm
    weight: 15
math: true
---

{{< note title="递归" >}}  

Leigh Caldwell在Stack Overflow上说的一句话: “如果使用循环，程序的性能可能更高;如果使用递归，程序可能更容易理解。如何选择要看什么对你来说更重要。”<br>

编写递归函数时，必须告诉它何时停止递归。正因为如此，每个递归函数都有两部分:
1. 基线条件(base case)。指的是函数不再调用自己，从而避免形成无限循环。
2. 递归条件(recursive case)。指的是函数调用自己。


{{< /note >}}


{{< note title="最大公约数" >}}

1. 原始：求两个数(N, M)的最大公约数。
2. 变形1：假设你是农场主，有一小块土地。你要将这块地均匀地分成方块，且分出的方块要尽可能大。

```
伪代码-思路：  
假设：N表示较小的数，M表示较大的数。  
重复一下操作，直到 N=0
新N = M % N
新M = 原N


```

{{< /note >}}


{{< note title="图的搜索" >}}  

> <font color=#a020f0>广度优先搜索</font>：可以回答两类问题，即：适合<font color=#a020f0>非加权图</font>
> 1. 从节点A出发，有往节点B的路径吗？
> 2. 从节点A出发，前往节点B的那条<font color=#a020f0>路径最短</font>。

---

> <font color=#a020f0>狄克斯特拉算法</font>：适合 <font color=#a020f0>没有负权边的加权图</font>。
> 1. 狄克斯特拉算法，假设：对于处理过的节点，没有前往该节点的更短路径。这种假设仅在<font color=#a020f0>没有负权边</font>时才成立。

---

> <font color=#a020f0>贝尔曼-福德算法</font>：适合 <font color=#a020f0>包含负权边</font>的加权图

---

狄克斯特拉算法包括4个步骤
> 1. 找出”最便宜“的节点，即：可在最短时间内到达的节点
> 2. 更新该节点的邻居的开销，检查是否有前往它们的更短路径，如果有，就更新其开销。
> 3. 重复这个过程，直到对图中的每个节点都这样做了
> 4. 计算最终路径

例如：乐谱 -换-> 钢琴
<p align="center"><img src="/datasets/note/dikesi.png" width="100%" height="100%" title="dikesi" alt="dikesi"></p>

**第一步**：找出最便宜的节点。这里，换海报最便宜了，不需要支付额外的费用。<br>
**第二步**：计算前往该节点的各个邻居的开销。
<p align="center"><img src="/datasets/note/dikesi-2.png" width="100%" height="100%" title="dikesi" alt="dikesi"></p>
父节点：代表该节点的上一级最便宜节点。<br>

**第三步**：目前条件(未遍历：<font color=#a020f0>黑胶唱片、吉他、架子鼓</font>；已遍历：海报)，在目前未遍历节点中找下一个最便宜的节点是 ”黑胶唱片“；更新 ”黑胶唱片“ 的各个邻居的开销。
<p align="center"><img src="/datasets/note/dikesi-3.png" width="100%" height="100%" title="dikesi" alt="dikesi"></p>
下一个最便宜的是 吉他，因此更新其邻居的开销：
<p align="center"><img src="/datasets/note/dikesi-4.png" width="100%" height="100%" title="dikesi" alt="dikesi"></p>
下一个最便宜的是 架子鼓，因此更新其邻居的开销：
<p align="center"><img src="/datasets/note/dikesi-5.png" width="100%" height="100%" title="dikesi" alt="dikesi"></p>

**第四步**：所有节点都已遍历完了，当前，我们直到最短路径的开销是35美元，但如何确定这条路径呢？为此，可以根据父节点寻找。

{{< /note >}}

{{< note title="狄克斯特拉算法：python实例：乐谱 -换-> 钢琴" >}}  
```python
class Solution(object):
    def __init__(self):
        pass
    
    @staticmethod
    # 在未处理的节点中找出开销最小的节点
    def find_lowest_cost_node(costs, processed):
        lowest_cost = float('inf')
        lowest_cost_node = None
        for node in costs:
            cost = costs[node]
            if cost < lowest_cost and node not in processed:
                lowest_cost = cost
                lowest_cost_node = node

        return lowest_cost_node

    def dikesi(self, graph):

        # 开销-散列表。未知节点的开销，先设置为无穷大
        costs = {
          'A': 0, 
          'B': 5, 
          'C': 0, 
          'D': float('inf'), 
          'E': float('inf'), 
          'F': float('inf')
        }
        # 父节点-散列表
        parents = {'B': 'A', 'C': 'A', 'F': None}
        # 已处理过的节点
        processed = []

        node = 'A'
        while node is not None:
            cost = costs[node]
            neighbors = graph[node]
            # 遍历当前节点的所有邻居
            for n in neighbors:
                new_cost = cost + neighbors[n]
                # 如果当前节点前往邻居更近，就更新该邻居的开销；同时更新该邻居的父节点
                if costs[n] > new_cost:
                    costs[n] = new_cost
                    parents[n] = node
            processed.append(node)
            node = self.find_lowest_cost_node(costs, processed)

        return costs['F']

# 图-散列表
graph = {}
graph['A'] = {'B': 5, 'C': 0}
graph['B'] = {'D': 15, 'E': 20}
graph['C'] = {'D': 30, 'E': 35}
graph['D'] = {'F': 20}
graph['E'] = {'F': 10}
graph['F'] = {}
alpha = Solution()
rst = alpha.dikesi(graph)
```
{{< /note >}}

<!--
{{< note title="标题" >}}  
{{< /note >}}
-->
