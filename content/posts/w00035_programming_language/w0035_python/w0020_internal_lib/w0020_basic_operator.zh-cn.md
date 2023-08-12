---
title: "基础操作"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: 基础操作
    identifier: python-basic_operator
    parent: python-internal
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","基础操作"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数据类型与操作

|操作|说明||
|:---|:---|:---|
|`del A[i]`|删除列表A中下标为i的元素，其后的每个元素都前移一个位置|列表-删除|
|`A.pop()`|弹出列表尾部元素，相当于出栈|列表-删除|
|`A.pop(i)`|弹出列表中任何位置出的元素|列表-删除|
|`A.remove('a')`|有时候不知道索引号，只知道要删除的值；remove只删除第一个指定的值|列表-删除|
|`A.sort(reverse=True)`|对列表A从大到小排序，列表A被永久改变|列表-排序|
|`B=sorted(A)`|排序后，A没有被改变|列表-排序|
|`A.reverse()`|A列表被永久的翻转了一下|列表-翻转|
||||
||||


## 二、`*和**的作用`
1. `*` 在函数定义/调用时的应用
    - 在函数定义时：`*`让python创建一个名为topping的空元组，并将收到的所有值封装在这个元组中。
    ```python
    def make_pizza(size, *topping):  # 定义
        ...
    ```
    - 在调用时：`*`操作符自动把参数列表拆开
    ```python
    toppings = ['nushroom', 'green peppers', 'extra cheese']
    make_pizza(size, *toppings)  # 调用
    ```

2. `**` 在函数定义/调用时的应用
    - 在函数定义时：`**` 让python创建一个名为user_info的空字典，并将收到的所有键值对都封装到这个字典中。
    ```python
    def build_profile(first, last, **user_info):  # 定义
        ...
    ```
    - 在调用时：`**` 操作符自动把参数字典拆开
    ```python
    user_infos = {}
    build_profile(first, last, **user_infos)  # 调用
    ```

## 三、


