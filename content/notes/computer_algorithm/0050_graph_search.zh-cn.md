---
title: 图的搜索
weight: 220
menu:
  notes:
    name: 图的搜索
    identifier: notes-algorithm-grapth-search
    parent: notes-algorithm
    weight: 50
math: true
---

<!--
{{< note title="标题" >}}  
{{< /note >}}
-->

{{< note title="狄克斯特拉算法(Dijkstra)" >}}  

<a href="http://c.biancheng.net/algorithm/dijkstra.html" target="blank">参考算法解析</a> <br>

> <font color=#a020f0>广度优先搜索</font>：可以回答两类问题，即：适合<font color=#a020f0>非加权图</font>
> 1. 从节点A出发，有往节点B的路径吗？
> 2. 从节点A出发，前往节点B的那条<font color=#a020f0>路径最短</font>。

---

> <font color=#a020f0>狄克斯特拉算法(Dijkstra)</font>：适合 <font color=#a020f0>没有负权边的加权图</font>。
> 1. 狄克斯特拉算法，假设：对于处理过的节点，没有前往该节点的更短路径。这种假设仅在<font color=#a020f0>没有负权边</font>时才成立。
> 2. 狄克斯特拉算法，是典型最短路径算法，用于计算一个结点到其他结点的最短路径。 它的主要特点是以起始点为中心向外层层扩展(广度优先搜索思想)，直到扩展到终点为止。

---

> <font color=#a020f0>贝尔曼-福德算法</font>：适合 <font color=#a020f0>包含负权边</font>的加权图

---

狄克斯特拉算法(Dijkstra)包括4个步骤
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


<!----------------------------------------------->
{{< note title="弗洛伊德算法(Floyd)" >}}
> 和Dijkstra算法一样，弗洛伊德(Floyd)算法也是一种用于寻找给定的加权图中顶点间最短路径的算法。该算法名称以创始人之一、1978年图灵奖获得者、斯坦福大学计算机科学系教授罗伯特·弗洛伊德命名。
> 
> 1. 弗洛伊德算法(Floyd)计算图中各个顶点之间的最短路径
> 2. 迪杰斯特拉算法用于计算图中某一个顶点到其他顶点的最短路径。
> 3. 弗洛伊德算法 VS 迪杰斯特拉算法：
>     1. 迪杰斯特拉算法通过选定的被访问顶点，求出从出<font color=#f00000>发访问顶点到其他顶点的最短路径</font>；
>     2. 弗洛伊德算法中每一个顶点都是出发访问点，所以需要将每一个顶点看做被访问顶点，求出从<font color=#f00000>每一个顶点到其他顶点的最短路径</font>。

---
{{< alert type=“info” >}}

> 弗洛伊德算法(Floyd) 的步骤：<br>
> 1. 定义初始化矩阵：距离矩阵 $D_0$、节点序列矩阵 $S_0$。
>    1. $d_{ij}$：表示 节点i 到 节点j 的最短距离。
>    2. $s_{ij}$：表示 节点i 到 节点j 需要经过节点j。
> 2. 一般经过k步，每步表示：如果包含节点k，从节点i 到 节点j 的最短距离是否会更短。如果更短就包含 节点k；否则 不包含。<br>对于矩阵 $D_{k-1}$（上一步完成后的矩阵），如果满足条件： $d_{ik} + d_{kj} < d_{ij}, i \ne k, j \ne k, i \ne j$，则进行下面的操作：
>    1. 用 $d_{ik} + d_{kj}$ 替换矩阵 $D_{k-1}$ 中的元素 $d_{ij}$，从而得到矩阵 $D_k$
>    2. 用 $k$ 替换矩阵 $S_{k-1}$ 中的元素 $s_{ij}$，从而得到矩阵 $S_k$
>    3. 令 $k = k + 1$，如果 $k = n+1$，即：每个节点都遍历过了。停止，否则重复上面操作

例如： 
<p align="center"><img src="/datasets/note/floyd.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

步骤1：初始化矩阵：距离矩阵 $D_0$、节点序列矩阵 $S_0$。
<p align="center"><img src="/datasets/note/floyd_0.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

迭代1：令k=1，表示：从 节点i 到 节点j，如果经过 节点1 中转，路径是否会缩短。比较：$d_{ik} + d_{kj}$ 与 $D_0$ 中[i][j]，经过比较，只有： $d_{23}, d_{32}$
<p align="center"><img src="/datasets/note/floyd_1.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

迭代2：修改了：$d_{14}, d_{41}$ <br>
<p align="center"><img src="/datasets/note/floyd_2.png" width="100%" height="100%" title="floyd" alt="floyd"></p>
迭代3：修改了：$d_{15}, d_{25}$ <br>
<p align="center"><img src="/datasets/note/floyd_3.png" width="100%" height="100%" title="floyd" alt="floyd"></p>
迭代4：修改了：$d_{15}, d_{23}, d_{25}, d_{32}, d_{35}, d_{51}, d_{52}, d_{53}$ <br>
<p align="center"><img src="/datasets/note/floyd_4.png" width="100%" height="100%" title="floyd" alt="floyd"></p>
迭代5：没有修改<br>
最后得到的矩阵为
<p align="center"><img src="/datasets/note/floyd_5.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

**个人理解**：
<p align="center"><img src="/datasets/note/floyd_6.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

1. 遍历 $k \in (所有节点)$ ：<font color=#f00000>每次以一个节点为桥梁，检测能不能连通两个节点（在已经遍历的节点基础上，可能会连接得更远）；或者检测是不是能使两个节点的路径变得更短。</font>
2. 遍历 $k \in (所有节点)$ ：其实先遍历那个无所谓，比如 $ 1  \rightarrow 2  \rightarrow 4  \rightarrow 5$ 与 $5  \rightarrow 4  \rightarrow 2  \rightarrow 1$ 每什么区别。


这两个矩阵包含了网络中任意两个节点最短路径的所有信息。比如
1. 从矩阵$D$中可以看出节点1到节点5的最短路径长度为12。
2. 从矩阵$S$中发现，节点1到节点5的中间节点是4; 从节点1到节点4的中间节点是2；从节点1到节点2，没有中间节点。

{{</alert>}}

{{< /note >}}


{{< note title="弗洛伊德算法(Floyd)-实例" >}}  
```python
# encoding: utf-8

class Solution(object):
    def calcEquation(self, graph):
        arr = list(graph.keys())
        # floyd 算法
        for k in arr:
            for i in arr:
                for j in arr:
                    if k == i or k == j or i == j:
                        continue
                    if k in graph[i] and j in graph[k]:
                        if j in graph[i]:
                            graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
                        else:
                            graph[i][j] = graph[i][k] + graph[k][j]

        return graph


def main():
    print('#===run a program with a main function===#')
    differ = Solution()
    graph = {
        "1": {"2": 3, "3": 10},
        "2": {"1": 3, "4": 5},
        "3": {"1": 10, "4": 6, "5": 15},
        "4": {"2": 5, "3": 6, "5": 4},
        "5": {"4": 4}
    }
    rst = differ.calcEquation(graph)
    print(rst)


if __name__ == '__main__':
    main()

# 结果
# {
# 	"1": {"2": 3, "3": 10, "4": 8, "5": 12}, 
# 	"2": {"1": 3, "3": 11, "4": 5, "5": 9}, 
# 	"3": {"1": 10 "2": 11, "4": 6, "5": 10,}, 
# 	"4": {"1": 8, "2": 5, "3": 6, "5": 4}, 
# 	"5": {"1": 12, "2": 9, "3": 10, "4": 4}
# }


```

{{< /note >}}


{{< note title="克鲁斯卡尔(Kruskal)算法" >}}

<a href="http://c.biancheng.net/algorithm/kruskal.html" target="blank">参考算法解析</a> <br>

最佳应用：修路问题：<br>
有北京有新增7个站点(A, B, C, D, E, F, G) ，现在需要修路把7个站点连通，各个站点的距离用边线表示(权) ，比如 A – B 距离 12公里，问：如何修路保证各个站点都能连通，并且总的修建公路总里程最短?
<p align="center"><img src="/datasets/note/kruskal.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

**克鲁斯卡尔(Kruskal)算法** ：用来求加权连通图的最小生成树的算法，采用了<font color=#f00000>贪心算法</font>
1. 基本思想：按照权值，从小到大的顺序选择n-1条边，并保证这n-1条边不构成回路
2. 具体做法：
    1. 将联通网中所有的边按照权值大小做升序排序，从权值最小的边开始选择，只要此边不构成回路，就可以选择它组成最小生成树
    2. 对N个顶点的联通网，挑选出 $N-1$ 条符合条件的边，这些边组成的生成树就是最小生成树。
    

{{< /note >}}


{{< note title="普利姆(Prim)算法" >}}
<a href="http://c.biancheng.net/algorithm/prim.html" target="blank">参考算法解析</a> <br>

普利姆(Prim)算法：查找最小生成树的过程，采用了贪心算法的思想，对于包含N个顶点的连通网，买次从连通网中找出一个权值最小的边，这样的操作重复 $N-1$ 次，由 $N-1$ 条权值最小的边组成的生成树，就是最小生成树。<br>

最佳应用：修路问题：<br>
有北京有新增7个站点(A, B, C, D, E, F, G) ，现在需要修路把7个站点连通，各个站点的距离用边线表示(权) ，比如 A – B 距离 12公里，问：如何修路保证各个站点都能连通，并且总的修建公路总里程最短?
<p align="center"><img src="/datasets/note/kruskal.png" width="100%" height="100%" title="floyd" alt="floyd"></p>

**思路**<br>
1. 将连通网中的所有顶点分为两类（假设为：A类、B类）。初始状态下，所有顶点位于B类；
2. 选择任意一个顶点，将其从B类移动到A类；
3. 从B类的所有顶点出发，找出一条连接着A类中某个顶点且权值最小的边，将次边连接的B类中的顶点移动到A类;
4. 重复执行第3步，直到B类中的所有顶点全部移动到A类，恰好可以找到 $N-1$ 条边。

例如：

<p align="center"><img src="/datasets/note/prim_1.png" width="100%" height="100%" title="prim" alt="prim"></p>

1. 初始化状态： $A = \lbrace \rbrace, B = \lbrace A, B, C, D, S, T \rbrace$
2. 随便选一个点，比如 $S$，从B类移动到A类：$A = \lbrace S \rbrace$ ，找与 $S$ 最近的点 是 $A$
3. 重复上面的操作，直到找到N-1个边

{{< /note >}}


{{< note title="马踏棋盘算法" >}}

1. 马踏棋盘算法也被称为骑士周游问题实际上是图的深度优先搜索(DFS)的应用
2. 将马随机放在国际象棋的8×8棋盘Board[0～7][0～7]的某个方格中，马按走棋规则(马走日字)进行移动。要求每个方格只进入一次，走遍棋盘上全部64个方格
<p align="center"><img src="/datasets/note/board.png" width="70%" height="70%" title="floyd" alt="floyd"></p>

{{< /note >}}



<!--
{{< note title="标题" >}}
{{< /note >}}
-->

