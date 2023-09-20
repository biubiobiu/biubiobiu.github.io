---
title: 滑动窗口
weight: 220
menu:
  notes:
    name: 滑动窗口
    identifier: notes-algorithm-sliding-window
    parent: notes-algorithm
    weight: 40
---

{{< note title="滑动窗口算法" >}}
<a href="https://www.cnblogs.com/huansky/p/13488234.html" target="blank">参考</a>


> <font color=#f00000>个人理解，滑动窗口主要解决的问题特点</font>：
> 1. 连续性，一定是连续序列或者字符串的最长/最短 的问题。


滑动窗口算法：是在给定特定窗口大小的数组或字符串上执行要求的操作，该技术可以将一部分问题中的嵌套循环转变为一个单循环，可以减少时间复杂度。即：在一个特定大小的字符串/数组上进行操作，而不是在整个字符串/数组上操作，这样就降低了问题的复杂度。<br>
滑动：说明这个窗口是移动的；<br>
窗口：窗口大小并不是固定的，可以不断扩容直到满足一定的条件；也可以不断缩小，直到找到一个满足条件的最小窗口；也可以是固定大小。<br>

滑动窗口算法的思路：<br>
1. 我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引闭区间 [left, right] 称为一个「窗口」。
2. 我们先不断地增加 right 指针扩大窗口 [left, right]，直到窗口中的字符串符合要求（包含了 T 中的所有字符）。
3. 此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right]，直到窗口中的字符串不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新一轮结果。
4. 重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。

`对于固定窗口大小，框架总结如下：`
```python
# 固定窗口大小为k
# 在s中 寻找窗口大小为k时的所包含最大元音字母个数
right = 0
while right<len(s):
  window.append(s[right])
  right += 1
  # 如果符合要求，说明窗口构造完成
  if right>=k:
    # 这已经是一个窗口了，根据条件做一些事情 ... 可以计算窗口最大值
    # 最后不要忘记把 【right-k】位置元素从窗口里移除
```

`对于不固定窗口大小，框架总结如下：`
```python
# 在s中寻找 t 的 最小覆盖子串
left, right = 0, 0
while right<len(s):
  right += 1
  # 如果符合要求，说明窗口构造完成，移动left缩小窗口
  while '符合要求':
    # 如果这个窗口的子串更短，则更新res
    res = minLen(res, windown)
    window.remove(left)
    left += 1
return res
```

{{< /note >}}

