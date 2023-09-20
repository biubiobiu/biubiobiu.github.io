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
|`ord()`|获取字符的ASCII码，比如：两个字符相减：ord('a') - ord('b')||
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

## 三、%的作用


1. %字符：标记转换说明符的开始
2. 转换标志：
   * - 表示左对齐；
   * + 表示在转换值之前要加上正负号；
   * ""(空白字符)表示正数之前保留空格；
   * 0表示转换值如果位数不够用0填充。
3. (.)前的数：最小字段宽度：转换后的字符串至少应该具有该值指定的宽度。
4. (.)后的数：精度值：
5. '{:.2f}'.format()，format的精度

```python
pi = 3.141592653

print('{:.2f}'.format(pi))
3.14

# 字段宽：10，精度：3
print('%10.3f' % pi)
     3.142
# 用*从后面的元组中读取字段宽度和精度
print('%.*f' % (3, pi))
3.142
# 用0填充空白
print('%010.3f' % pi)
000003.142
# 左对齐
print('%-10.3f' % pi)
3.142  
# 显示正负号
print('%+f' % pi)
+3.141593
```

## 四、类的继承

约定：
1. python中首字母大写的名称为类名
2. 类中的函数成为方法
3. 类中的变量成为属性

`__init__`：是一个特殊的方法，每当根据Dog类创建新实例时，python会自动运行它。其形参self是必不可少的，且必须在前面。
当python实例化对象时，会调用这个__init__()方法来创建Dog实例，自动传入实参self，self是一个指向实例本身的引用，每个与类相关联的方法调用都自动传递实参self。<br>

**类的继承：**<br>
super()，解决了子类调用父类方法的一些问题，父类多次被调用时，只执行一次。


```python
class Car():
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0
        
    def read_odoeter(self):
        print("")
        
class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery_size = 70
        
    def describe_battery(self):
        print("")

```

