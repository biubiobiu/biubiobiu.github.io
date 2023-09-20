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

<!--
{{< note title="标题" >}}  
{{< /note >}}
-->

<!--===============递归===============-->
{{< note title="递归" >}}  

Leigh Caldwell在Stack Overflow上说的一句话: “如果使用循环，程序的性能可能更高;如果使用递归，程序可能更容易理解。如何选择要看什么对你来说更重要。”<br>

编写递归函数时，必须告诉它何时停止递归。正因为如此，每个递归函数都有两部分:
1. 基线条件(base case)。指的是函数不再调用自己，从而避免形成无限循环。
2. 递归条件(recursive case)。指的是函数调用自己。

{{< /note >}}


<!--===============二分查找===============-->
{{< note title="二分查找" >}}  
比如：从1~100的数字中，我认选一个，让你猜。我只会说：大了、小了、对了。需要猜多少次呢？<br>
二分查找：一半一半的猜，每次都排除一半。所以需要的次数是：log<sub>2</sub>N。（向上取整）
<p align="center"><img src="/datasets/note/dichotomy.png" width="100%" height="100%" title="log2" alt="log2"></p>

```python
class BinarySearch(object):
    # 迭代
    def search_iterative(self, nums, item):
        low = 0
        high = len(nums) - 1
        while low<=high:
            mid = (low + high) // 2
            guess = nums[mid]
            if guess == item:
                return mid
            elif guess > item:
                high = mid - 1
            else:
                low = mid + 1
        return None
    
    # 递归
    def search_recursive(self, nums, low, high, item):
        if high >= low:
            mid = (high + low) // 2
            guess = nums[mid]
            if guess == item:
                return mid
            elif guess > item:
                return self.search_recursive(nums, low, mid-1, item)
            else:
                return self.search_recursive(nums, mid+1, high, item)
        else:
            return None

```

---
<a href="https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-interview-150" target="blank">在排序数组中查找元素的第一个和最后一个位置
</a> <br>
体会一下：二分查找法。如下代码中：nums为有序数组，ans 在 `nums[mid] > target` 中跟新，即：<font color=#f00000>切分到最后，ans记录的是第一个 大于 target的值。</font>

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)

        def binary_search(lower):
            left, right, ans = 0, n-1, n 

            while left <= right:
                mid = (left + right)//2
                if nums[mid] > target or (lower and nums[mid] >= target):
                    right = mid-1
                    ans = mid
                else:
                    left = mid + 1

            return ans

        left_id = binary_search(True)
        right_id = binary_search(False)-1

        if left_id <= right_id and right_id < n and nums[left_id] == target and nums[right_id]==target:
            return [left_id, right_id]
        return [-1, -1]
```

---

问题描述：给两个有序数组，nums1、nums2。返回 这两个有序数按照顺序组合并后的第k个值。<br>
这个可以用来解决：<a href="https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=study-plan-v2&envId=top-interview-150" target="blank">寻找两个正序数组的中位数
</a>
```python
nums1 = [1, 2, 5, 9]
nums2 = [3, 4, 6, 8, 13]
m, n = len(nums1), len(nums2)

def get_k(k):
    idx1, idx2 = 0, 0
    while True:
        # 特殊情况：其中一个数组为空
        if idx1 == m:
            return nums2[idx2+k-1]
        if idx2 == n:
            return nums1[idx1+k-1]
        if k == 1:
            return min(nums1[idx1], nums2[idx2])
        # 从0开始，比较两个数组 第k//2位置的大小；然后调整 分割线：idx1, idx2和k的值
        new_idx1 = min(idx1 + k // 2 - 1, m - 1)
        new_idx2 = min(idx2 + k // 2 - 1, n - 1)
        pivot1, pivot2 = nums1[new_idx1], nums2[new_idx2]
        # 如果nums1[new_idx1]位置的值比较小，说明这 new_idx1+1个数，肯定是较小的值排在前面
        # 所以：起始位置 idx1 移动到new_idx1+1，k = k - (new_idx1-idx1+1)，即：删掉了new_idx1-idx1+1个元素
        # 从寻找第k个值，变成：寻找第k-(new_idx1-idx1+1)个值
        if pivot1 <= pivot2:
            k -= new_idx1 - idx1 + 1
            idx1 = new_idx1 + 1
        else:
            k -= new_idx2 - idx2 + 1
            idx2 = new_idx2 + 1
```


{{< /note >}}


<!--===============最大公约数===============-->
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


<!--===============摩尔投票算法===============-->
{{< note title="摩尔投票算法" >}}
**问题描述**：给定一个大小为n的数组，其中有一个元素，出现次数大于 n/2。只遍历一次数组，找出这个元素。<br>
1. 这个元素就是 “众数”。
2. 投票：众数：票数+1，非众数：票数-1。
3. 所以，用 票数(cnt) 和 相应的元素对应起来。如果cnt=0时，就更换元素。


{{< /note >}}


<!--===============数组的翻转-旋转===============-->
{{< note title="数组的翻转-旋转" >}}
**数组的旋转**：移动k次，从1 2 3 4 5 6 --移动k次-> 5 6 1 2 3 4 <br>
**直接 环状替换:**
<p align="center"><img src="/datasets/note/list_rotate.png" width="100%" height="100%" title="list_rotate" alt="list_rotate"></p>

1. 从0位置开始，替换到下一个，下一个继续替换到下一个，直到回到0位置。
2. 回到0位置后，是否遍历完全部的数据呢？可以用一个count变量来记录已遍历的个数，直到遍历完毕。

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k = k%len(nums)
        if len(nums) < 2:
            return nums
        anchor, point =0, 0

        anchor_value = nums[anchor]
        for _ in range(len(nums)):
            anchor = (anchor + k) % len(nums)
            anchor_value, nums[anchor] = nums[anchor], anchor_value

            if anchor == point:
                point += 1
                anchor = point
                if anchor < len(nums):
                    anchor_value = nums[anchor]

        return nums
```

**数组的多次翻转==旋转**
|操作|结果|
|:--|:--|
|原始数组|1 2 3 4 5 6 7|
|翻转所有元素|7 6 5 4 3 2 1|
|翻转[0, k]区间|5 6 7 4 3 2 1|
|翻转[k, n]区间|5 6 7 1 2 3 4|

{{< /note >}}


<!--===============图搜索===============-->
{{< note title="图的搜索" >}}  

> <font color=#a020f0>广度优先搜索</font>：可以回答两类问题，即：适合<font color=#a020f0>非加权图</font>
> 1. 从节点A出发，有往节点B的路径吗？
> 2. 从节点A出发，前往节点B的那条<font color=#a020f0>路径最短</font>。

---

> <font color=#a020f0>狄克斯特拉算法(Dijkstra)</font>：适合 <font color=#a020f0>没有负权边的加权图</font>。
> 1. 狄克斯特拉算法，假设：对于处理过的节点，没有前往该节点的更短路径。这种假设仅在<font color=#a020f0>没有负权边</font>时才成立。

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

<!--=============双指针==============-->
{{< note title="双指针" >}}  

示例：<a href="https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-interview-150" target="blank">盛更多的水</a>
<p align="center"><img src="/datasets/note/double_point.png" width="100%" height="100%" title="double" alt="double"></p>

通过这个题目，来体会一下 <font color=#f00000>双指针</font>的优雅。<br>
1. 开始的状态：一个指针（left）指向 头，一个指针（right）指向 尾；
2. 当 $h[left_0] < h[right_0]$时，则把 $left_0+=1$ 。<br>
   这是因为：当 $h[left_0] < h[right_0]$ 时，如果 $left_0$ 指针不动，调整 $right_0$ 指针， $ right \in (left_0, right_0]$  这个系列的值对会小于 起始值，所以，<font color=#f00000>相当于这个系列拿最大值比较，其他的就不用考虑了。</font>
3. 所以，每个调整小边。相当于每次过滤了一批可能项。对比两个for循环的话，所有可能项都会遍历。

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        max_c = 0
        n = len(height)
        i, j = 0, n-1
        while i < j:
            if height[i] < height[j]:
                max_c = max(max_c, height[i] * (j-i))
                i += 1
            else:
                max_c = max(max_c, height[j] * (j - i))
                j -= 1
        return max_c
```


{{< /note >}}


<!---------------------固定位数-比较---------------------->
{{< note title="固定位数-比较" >}}  
示例：<a href="https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-interview-150" target="blank">字母异位词分组</a> <br>
**关键点**：26个字母是固定的，异位词：相同字母的个数是一样的，就是位置可能不一样。每个字符串 都可以用 [0]*26，累积每个位置字母的个数。用这个 元组作为比较的可以。 <br>

类似这个题目，<font color=#f00000>有固定位数的情况，可以考虑 固定位数的数组（[0]*26）、固定位数的二进制表示。</font>


{{< /note >}}

<!---------------------KMP算法---------------------->
{{< note title="KMP算法" >}}
KMP算法，是经典的字符匹配算法。<br>
在介绍KMP之前，先介绍 字符串的 <font color=#f00000>前缀集合</font> 和 <font color=#f00000>后缀集合</font> <br>
比如：abab，前缀集合：a, ab, aba； 后缀集合：b, ab, bab。<br>

KMP算法的思路：
1. 逐个 原字符串 和 匹配字符串。每次匹配失败时，就不用从0开始，而是从当前 <font color=#f00000>前/后缀 的最长长度</font> 开始
2. 用next记录 匹配字符串 的每个位置的最长 前/后缀 的最长长度


<p align="center"><img src="/datasets/note/kmp_0.jpg" width="100%" height="100%" title="KMP" alt="KMP"></p>

在计算next数组时，当匹配失败时，为什么是：$ j = next[j-1] $ ？
1. 此时，$p[j] \ne p[i]$。$p[:j]$：表示目前最长的前缀
2. $next[j-1]$ ：表示 在 $p[:j]$ 字符串中，最长的 前/后缀长度，假设 $ k = next[j-1] $。
3. 所以 $ p[:k] == p[i-k: i] == p[j-k: j] $，这三段字符串相等，所以，$ j = next[j-1] $，即：从 $ p[k] $ 开始 继续与 $p[i]$ 比较


```python
def get_next(p):
    n = len(p)
    # i: 表示遍历的字符下标
    # j: 最大前后缀长度
    i, j = 1, 0
    # 记录：每个当前的最大前后缀长度
    next = [0] * n

    for i in range(1, n):
        # 如果相等，长度+1
        if p[i] == p[j]:
            j += 1
            next[i] = j
        # 如果不相等，更新长度的值，直到相等或者长度为0
        else:
            while j > 0 and p[i] != p[j]:
                j = next[j-1]
    return next


def kmp_search(string, patt):
    # 获取next数组
    next = get_next(patt)
    i, j, n, m = 0, 0, len(string), len(patt)
    while i < n and j < m:
        if string[i] == patt[j]:
            i += 1
            j += 1
        elif j > 0:   # 字符匹配失败，根据next跳过子串前面的一些字符
            j = next[j-1]
        else:  # 子串第一个字符就失配
            i += 1
        if j == m:
            return i-m

    return -1
```

{{< /note >}}

<!--
{{< note title="标题" >}}  
{{< /note >}}
-->
