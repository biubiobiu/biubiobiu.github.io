---
title: 五大常用算法
weight: 220
menu:
  notes:
    name: 五大常用算法
    identifier: notes-algorithm-dynamic
    parent: notes-algorithm
    weight: 30
math: true
---


<!--=================分治法=================-->

{{< note title="分治法" >}} 
分治法(divide and conquer)的工作原理：
1. 找出简单的基线条件。
2. 确定如何缩小问题的规模，使其符合基线条件。

分治法：并非可用于解决问题的算法，而是一种解决问题的思路。

1. 待解决复杂问题，能够简化为若干个小规模相同的问题，<font color=#a020f0>各个子问题独立存在</font>，并且与原问题形式相同；
2. 递归地解决各个子问题；
3. 将各个子问题的解合并，得到原问题的解。

**实例1：**<br>
N和M的最大公约数（把一块农田均分成方块，求方块最大值）

**实例2:**<br>
快速排序：
1. 基线条件：空数组或者只有一个元素的数组，直接返回
2. 选中一个基准值后，小于基准值的放在左边，大于基准值的放在右边。

<p align="center"><img src="/datasets/note/quik_sort.png" width="100%" height="100%" title="merge_sort" alt="merge_sort"></p>

```python
def quick_sort(nums):
    if len(nums) < 2:
        return nums
    else:
        pivot = nums[0]
        lesser, greater = [], []
        for item in nums[1:]:
            if item <= pivot:
                lesser.append(item)
            else:
                greater.append(item)
        return quick_sort(lesser) + [pivot] + quick_sort(greater)

```

**实例3：**<br>
归并排序
<p align="center"><img src="/datasets/note/merge_sort.png" width="100%" height="100%" title="merge_sort" alt="merge_sort"></p>

**实例4：**<br>
数组中最大值<br>
1. 普通的做法：设置个变量记录当前的最大值，变量数组中的每个值，最终找到数组的最大值。这中做法的时间复杂度 $O(n)$
2. 分治法：把数组分成两半：分别找打这两个子数组的最大值，再从这两个值中选出最大值。以此类推。这种做法的时间复杂度 $O(\log n)$

{{< /note >}}



<!--=================贪心算法=================-->
{{< note title="贪心算法" >}}

> **近似算法**：approximation algorithm. 在获得精确解需要的时间太长时，可使用近似算法。判断近似算法优劣的标准如下：
> 1. 速度有多快
> 2. 得到的近似解与最优解的接近程度

---
> <font color=#a020f0>NP完全问题</font>：就是以难解著称的问题。很多非常聪明的人都认为，根本不可能编写出可快速解决这些问题的算法。
> 1. <kbd>集合覆盖问题</kbd>：有n个广播站，每个广播站可能覆盖几个省(覆盖有重复)，想要覆盖全国，最少需要选那几个广播站。每个广播站覆盖的范围：是一个集合。想要全集：选最少个集合，并集是全集。
> 2. <kbd>旅行商问题</kbd>：旅行商打算旅行n个城市，找出前往这n个城市的最短路径。如果要找最优解：有<mark>n!</mark>种可能。


> <font color=#a020f0>如何识别NP完全问题</font>：如果能够判断是NP完全问题，这样就好了，就不用去寻找完美的解决方案，而是使用近似算法即可。
> 1. 元素较少时算法的运行速度非常快，但随着元素数量的增加，速度会变得非常慢
> 2. 涉及<font color=#a020f0>所有组合</font>的问题，通常是NP完全问题
> 3. 不能将问题分成小问题，必须考虑各种可能的情况，这<font color=#a020f0>可能是NP完全问题</font>
> 4. 如果问题涉及<font color=#a020f0>序列(比如：旅行商问题中的城市序列) 且难以解决</font>，可能就是NP完全问题
> 5. 如果问题涉及<font color=#a020f0>集合(比如：广播台集合) 且难以解决</font>，可能就是NP完全问题
> 6. 如果问题<font color=#a020f0>可转换为集合问题、旅行商问题</font>，它肯定就是NP完全问题。

---
> <font color=#a020f0>贪心算法</font>：
> 1. 贪心算法，是寻找局部最优解，企图以这种方式获得全局最优解
> 2. 面对<kbd>NP完全问题</kbd>，还没有找到快速解决方案。最佳的做法是使用<kbd>近似算法</kbd>
> 3. 贪心算法，是一种易于实现、运行速度快 的近似算法。


> 贪心算法解决的问题：
> 1. 教室，安排课程
> 2. 旅行商问题
> 3. 序列全排列问题
---

**贪心算法**：在对问题求解时，总是做出在当前看来是做好的选择。即：<font color=#a020f0>当考虑做何种选择的时候，我们只考虑对当前问题最佳的选择而不考虑子问题的结果，这是贪心算法可行的第一个基本要素</font>。不从整体最优上考虑，而是仅仅在某种意义上的局部最优解。贪心算法以迭代的方式作出相继的贪心选择，每做一次贪心选择就将问题简化为规模更小的子问题。

**何时采用贪心算法**：对于一个具体问题，要确定它是否具有贪心选择性质，必须证明<font color=#a020f0>每一步所作的贪心选择最终导致问题的整体最优解。</font>

**示例：**<br>
完全背包问题、均分纸牌、最大整数

实际上，贪心算法适用的情况很少。需要先证明：<font color=#a020f0>局部最优解会得出整体最优解</font>，才可以使用。一旦证明能成立，它就是一种高效的算法。<br>
例如【0-1背包问题】：即：对于每个物品，要么装要么不装(0或1)<br>
有一个背包，背包容量是M=150。有7个物品，物品可以分割成任意大小。要求尽可能让装入背包中的物品总价值最大，但不能超过总容量。<br>
物品： A B C D E F G<br>
重量： 35 30 60 50 40 10 25<br>
价值： 10 40 30 50 35 40 30<br>
目标函数： ∑pi最大<br>

利用贪心算法，可以这样：
1. 每次挑选价值最大的物品装入背包，（是否是最优解？）
2. 每次选择重量最小的物品装入背包，（是否是最优解？）
3. 每次选择单位重量价值最大的物品，（是否是最优解？）

上面的3中贪心策略，都无法成立，所以不能采用贪心算法。所以，<font color=#a020f0>贪心算法虽然简单高效，但是能证明可以使用该算法的场景比较少。</font>

{{< /note >}}


<!--=================动态规划=================-->
{{< note title="动态规划" >}}  
**与分治法的不同**：<br>
动态规划与分治法相似，都是组合子问题的解来解决原问题，与分治法的不同在于：
1. 分治法：将原问题划分为一个个<font color=#a020f0>不相交</font>的子问题（比如：归并排序，将数组不断地划分为一个个的子数组进行排序，再将返回的两个有序数组进行合并排序）
2. 动态规划：要解决的是<font color=#a020f0>子问题有重叠</font>的问题，例如0-1背包问题。即：不同的子问题有公共的子子问题，这些重叠的子问题在动态规划中是不应该也不需要重新计算的，而是应该将其解以一定方式保存起来，提供给父问题使用。


**哪些可以使用动态规划呢？**<br>
1. 动态规划可以帮助你在<font color=#a020f0>给定约束条件</font>下找到最优解。比如：在背包问题中，你必须在背包容量给定的情况下，偷取价值最高的商品。
2. 在问题可以分解为<font color=#a020f0>彼此独立且离散</font>的子问题时，可以使用动态规划来解决问题

**怎么设计动态规划？**：<br>
要设计出动态规划解决方案可能很难，下面给出一些小贴士：
<font color=#a020f0>
1. 每种动态规划解决方案都涉及网格<br>
2. 单元格的值通常就是你需要优化的值。
3. 每个单元格都是一个子问题，因此你应考虑如何将问题分成子问题，这有助于找出网格的坐标轴。

</font>

**个人理解**：
1. 在设计网格时，要考虑：限制条件、分解子问题。
2. 每个单元格是一个需要优化的值，这个值是在 限制条件下的子问题的最优解。如果限制条件不一样，分解的子问题也会不一样。所以：<font color=#a020f0>限制条件</font>要找准。

<!--
<font color=#a020f0></font>
-->

**例如**:
1. 最长公共子串(要求连续相同)： 在设计网格时，每个单元格是一个需要优化的值，含义是：公共字符串必须含有当前位置字符结尾的情况下的最长公共子串。所以才会有：<font color=#a020f0>如果结尾不相同，值=0；如果结尾相同，左上角 + 1。</font> 
2. 最长公共子序列(不要求连续相同，只要顺序一致上 相同)：在设计网格时，每个单元格是当前状态下最长公共子序列的长度值。所以，在两个字母不同时会保持当前最大值。<font color=#a020f0>如果两个字母不同，max(上方，左边)；如果两个字母相同，左上方 + 1</font>。 如果是 max(上边，左边) + 1，会是怎么呢？答案是：如果有多个相同的，会累积。

<p align="center"><img src="/datasets/note/long_string.png" width="100%" height="100%" title="long_string" alt="long_string"></p>

**设计步骤**：<br>
动态规划通常用来求解<font color=#a020f0>最优解问题</font>，这类问题会有很多个解，每个解都对应一个值，而我们则希望在这些解中找到最优解（最大值或者最小值）。
通常四个步骤设计一个动态规划算法：
1. 定义dp数组以及下标的含义；
2. 推导出：<font color=#a020f0>递推公式</font>
3. dp数组的初始化
4. 遍历顺序
5. 打印出dp数组

**实现方法**：
1. 递归，属于自顶向下的计算方法：如果子问题有重复计算的情况下，需要一个<font color=#a020f0>备忘录</font>来辅助实现，<font color=#a020f0>备忘录</font>主要用来保存每一个子问题的解，当每个子问题只求一次，如果后续需要子问题的解，只需要查找备忘录中保存的结果，不必重复计算。
2. 动态规划，属于自底向上的计算方法：此方法最常用，必须明确每个子问题规模的概念，使得任何子问题的求解都依赖于子子问题的解来进行求解。

**示例：**<br>
<a href="https://www.bilibili.com/read/cv12924751" target="blank">0-1背包问题</a> <br>
最长公共子串<br>
最长公共子序列<br>
<a href="https://leetcode.cn/problems/edit-distance/?envType=study-plan-v2&envId=top-interview-150" target="blank">编辑距离</a> <br>

---
**股票问题** ： 所有股票问题都是要<font color=#f00000>最大化手里持有的钱</font>。 买股票手里的钱减少，卖股票手里的钱增加，无论什么时刻，我们要保证手里的钱最多。而且，本次买还是卖只跟上一次我们卖还是买的状态有关。 
1. buy和sell都代表操作之后手里的钱。
2. buy和sell：都有两个状态，操作、不操作。所以，当buy和sell 保持上一次的状态，就表示本次没有买卖操作；当buy和sell状态有变动，就表示本次有买卖操作。具体是否需要有操作，是优化条件决定的：<font color=#f00000>保持手里的钱最多。</font>

**实例**
* 只交易1次<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/" target="blank">121. 买卖股票的最佳时机
</a> 
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            buy, sell = -float("inf"), 0
            for p in prices:
                buy = max(buy, 0 - p)
                sell = max(sell, buy + p)
            return sell
    ```
* 交易无限次<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/" target="blank">122. 买卖股票的最佳时机 II</a> <br>
这两个问题唯一的不同点在于我们是买一次还是买无穷多次，而代码就只有 0-p 和 sell-p 的区别。 因为如果买无穷多次，就需要上一次卖完的状态。如果只买一次，那么上一个状态一定是0。
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            
            buy, sell = -float("inf"), 0

            for p in prices:
                buy = max(buy, sell - p)
                sell = max(sell, buy + p)

            return sell
    ```
* 只交易2次<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/" target="blank">123. 买卖股票的最佳时机 III</a> 
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:

            b1, b2, s1, s2 = -float("inf"), -float("inf"), 0, 0

            for p in prices:
                b1 = max(b1, 0 - p)
                s1 = max(s1, b1 + p)
                b2 = max(b2, s1 - p)
                s2 = max(s2, b2 + p)
                
            return s2
    ```
* 只交易k次<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/" target="blank">188. 买卖股票的最佳时机 IV</a> 
    ```python
    class Solution:
        def maxProfit(self, k: int, prices: List[int]) -> int:
            k = min(k, len(prices) // 2)

            buy = [-float("inf")] * (k+1)
            sell = [0] * (k+1)

            for p in prices:
                for i in range(1, k+1):
                    buy[i] = max(buy[i], sell[i-1] - p)
                    sell[i] = max(sell[i], buy[i] + p)

            return sell[-1]
    ```
* 可交易无限次 + 冷冻期<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/" target="blank">309. 买卖股票的最佳时机含冷冻期</a> <br>
这道题只是第二题的变形，卖完要隔一天才能买，那么就多记录上一次卖的状态即可。
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:

            buy, sell_pre, sell = -float("inf"), 0, 0

            for p in prices:
                buy = max(buy, sell_pre - p)
                sell_pre, sell = sell, max(sell, buy + p)
                    
            return sell
    ```
* 可交易无限次 + 手续费<br>
<a href="https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/" target="blank">714. 买卖股票的最佳时机含手续费</a> <br>
每次买卖需要手续费，那么我们买的时候减掉手续费就行了。
    ```python
    class Solution:
        def maxProfit(self, prices: List[int], fee: int) -> int:

            buy, sell = -float("inf"), 0

            for p in prices:
                buy = max(buy, sell - p - fee)
                sell = max(sell, buy + p)
            
            return sell

    ```



{{< /note >}}


<!--=================回溯法=================-->
{{< note title="回溯法" >}}
**回溯法：**
是一种类似枚举的搜索尝试过程，在搜索尝试过程中寻找问题的解，当发现已不满足条件时，就`回溯`返回，尝试别的路径。<br>
回溯法是一种选优搜索法，通常是创建一棵树，从根节点出发，按照`深度优先搜索`的策略进行搜索，到达某一节点后，搜索该节点是否包含该问题的解：
- 设计状态：表示求解问题的不同阶段，在回溯的时候，要有`状态重置`
- 如果包含，则进入下一个节点进行搜索；
- 如果不包含，则`回溯`到父节点选择其他支路进行搜索。

**何时采用回溯算法：** 必须有标志性操作——`搜索时不满足条件就剪枝 + 所有解`

**设计步骤**:<br>
1. 针对所给的原问题，定义问题的解空间，设计状态，用于记录不同阶段
2. 确定易于搜索的解空间结构；
3. 以深度优先搜索解空间，并在搜索过程中用剪枝函数除去无效搜索。

**示例：**<br>
`全排列`、旅行商问题、八皇后问题<br>
例如：`全排列`
<p align="center"><img src="/datasets/note/permute.png" width="100%" height="100%" title="全排列" alt="全排列"></p>

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        visited = []
        rst = []
        n = len(nums)

        def combin(idx):
            if idx == n:
                rst.append(visited[:])
                return

            for num in nums:
                if num not in visited:
                    # 更新状态
                    visited.append(num)
                    combin(idx+1)
                    # 回溯
                    visited.pop()

        combin(0)

        return rst
```

---
对比这两种组合：<br>
1. <a href="https://leetcode.cn/problems/combinations/?envType=study-plan-v2&envId=top-interview-150" target="blank">组合</a>
    * $C_n^k$ : 排列组合，从n中选k个数，有多少种形式。
    * 选中一个数的情况，在剩下的选k-1个：`combin(idx+1, k-1)`
    * 不选中这个数的情况，在剩下的选k个：`combin(idx+1, k)`
2. <a href="https://leetcode.cn/problems/combination-sum/solutions/406516/zu-he-zong-he-by-leetcode-solution/?envType=study-plan-v2&envId=top-interview-150" target="blank">和的组合</a>
    * 元素可以<font color=#f00000>无限随便用</font>，只要最终和为k，有多少种形式。
    * 选中一个数 $a$ 的情况（累加 $a$），从这个数的下标开始，和 变成成了 $k-a$：`combin(idx, k-a)`
    * 不用这个数 $a$ 的情况（不在累加 $a$ 了），从这个数的下一个数开始，和 还是 $k$: `combin(idx+1, k)`
    * 可以理解为：每个数都会：累积1、2、3、4、。。。。次，直到触发终止条件。<br>

--- 

**实例** <br>
1. <a href="http://c.biancheng.net/algorithm/maze-puzzle.html" target="blank">迷宫问题</a>
2. <a href="http://c.biancheng.net/algorithm/n-queens.html" target="blank">N皇后问题</a> <br>N 皇后问题源自国际象棋，所有棋子中权力最大的称为皇后，它可以直着走、横着走、斜着走（沿 45 度角），可以攻击移动途中遇到的任何棋子。<br>N 皇后问题的具体内容是：如何将 N 个皇后摆放在 N*N 的棋盘中，使它们无法相互攻击。<br>
回溯算法解决N皇后问题的具体思路是：将 N 个皇后逐一放置在不同的行，以“回溯”的方式逐一测试出每行皇后所在行的具体位置，最终确定所有皇后的位置。


```python
# 伪代码
输入 N      // 输入皇后的个数
q[1...N]    //存储每行的皇后的具体位置（列标）
n_queens(k , n):    // 确定第 k 行皇后的位置
    if k > n:             // 递归的出口
        Print q          // 输出各个皇后的位置
    else:
        for j <- 1 to n:      // 从第 k 行第 1 列开始，判断各个位置是否可行
            if isSafe(k , j):    // 如果可行，继续判断下一行
                q[k] <- j        // 将第 k 行皇后放置的位置 j 记录下来
                n_queens(k+1 , n)    // 继续判断下一行皇后的位置

# python代码
class Solution:
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def generateBoard():
            board = list()
            for i in range(n):
                row[queens[i]] = "Q"
                board.append("".join(row))
                row[queens[i]] = "."
            return board

        def backtrack(row):
            if row == n:
                board = generateBoard()
                solutions.append(board)
            else:
                for i in range(n):
                    if i in columns or row-i in diagonal1 or row+i in diagonal2:
                        continue
                    queens[row] = i 
                    columns.add(i)
                    diagonal1.add(row+i)
                    diagonal2.add(row-i)
                    backtrack(row+1)
                    diagonal2.remove(row-i)
                    diagonal1.remove(row+i)
                    columns.remove(i)
        
        solutions = list()
        queens = [-1] * n
        # 列：记录在列方向上是否已经放置
        columns = set()
        # 斜线1：从左上到右下方向：同一条斜线上的每个位置满足：行值 - 列值 是相等的。
        # 因此：使用 行值 - 列值 表示一条方向的斜线。
        diagonal1 = set()
        # 斜线2：从右上到左下方向：同一条斜线上的每个位置满足：行值 + 列值 是相等的。
        # 因此：使用 行值 + 列值 表示一条方向的斜线
        diagonal2 = set()
        row = ["."] * n
        backtrack(0)
        return solutions

```

{{< /note >}}


<!--=================分支限界法=================-->
{{< note title="分支限界法" >}}
**分支限界法(branch and bound method)：** 和回溯法类似，也是一种搜索算法，与回溯法不同的是：
1. 回溯法：找出问题的许多解；通常用`深度优先`的方式搜索解空间树；
2. 分支限界法：找出原问题的一个解，或者 在满足约束条件的解中找出使某一目标函数的极大解/极小解。通常以`广度优先或最小耗费优先`的方式搜索解空间树。

在当前节点(`扩展节点`)处，生成其所有的子节点(分支)，然后再从当前节点的子节点表中选择下一个`扩展节点`。为了有效地选择下一个`扩展节点`，加速搜索的进程，在每个节点处，计算一个`限界`，从其子节点表中选择一个最有利的节点作为`扩展节点`，使搜索朝着解空间上最优解的分支推进。

**何时采用分支界限法：** 必须有标志性操作——`搜索时不满足限界就剪枝 + 最优解`

**示例：**<br>
0-1背包问题：`限界`就是背包的大小，一个节点的子节点表中，如果有超过`限界`的就直接剪枝。如下图所示：
<p align="center"><img src="/datasets/note/branch_bound.jpg" width="100%" height="100%" title="branch_bound" alt="branch_bound"></p>

{{< /note >}}
