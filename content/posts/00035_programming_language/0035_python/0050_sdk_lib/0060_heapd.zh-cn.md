---
title: "堆-heapq"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: 堆-heapq
    identifier: python-sdk-heapq
    parent: python-sdk
    weight: 60
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","堆"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、堆

```python
import heapq
```

|操作|解释|功能|
|:--|:--|:--|
||例如：arr=[2, 9, 1, 4]||
|heapq.heapify(arr)|建堆，对列表arr建堆。<br>也可以这样：<br>arr = [(5, 'a'), (2, 'b'), (8, 'c'), (9, 'd'), (6, 'e'), (1, 'f')]<br>heapq.heapify(arr)  <br>然后arr就变成：<br>[(1, 'f'), (2, 'b'), (5, 'a'), (9, 'd'), (6, 'e'), (8, 'c')]<br>|建堆|
|heapq.heappush(arr, 10)|添加元素，然后再向上调整堆。例如：在arr列表中添加5，然后在维持一个堆|添加|
|heapq.heappop(arr, 10)|提取堆顶，然后把堆尾放在堆顶，最后对堆顶做向下调整。把arr的堆顶元素提取出来。|pop|
|heapq.heappushpop(arr, 10)|用新元素与堆顶做比较，如果堆顶大于新元素，直接返回新元素。否则返回堆顶，并把新元素放在堆顶后向下调整||
|heapq.heapreplace(arr, 10)|返回堆顶，再把新元素放在堆顶，然后对堆顶做向下调整。|替换|
|heapq.merge([], [], [])|多路归并：把排好序的多个list， 归并成一个list<br>例如：list(heapq.merge([1,3,4], [2,3,9], [5,8], reverse=False))|多路归并|
|heapq.nlargest(n,iterable)|结果上等价于：soted(iterable, key=key, reverse=True)[:n]<br>从一段数据上获取最大的n个数|前n个最大值|
|heapq.nsmallest(n,iterable)|结果上等价于：soted(iterable, key=key, reverse=False)[:n]<br>从一段数据上获取最小的n个数|前n个最小值|

## 二、



