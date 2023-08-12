---
title: "importlib包"
date: 2021-12-08T06:00:20+08:00
menu:
  sidebar:
    name: importlib
    identifier: python-sdk-importlib
    parent: python-sdk
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","配置"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## import_module()函数
背景：一个函数运行，需要根据不同项目的配置，动态导入对应的`配置文件`。
例如：如下路径，向a模块中导入c.py中的对象
a  
├── a.py  
├── `__`init`__`.py  

b  
├── b.py  
├── c    
│　　├── c.py　　　　# 该文件中，有变量args=[]，class C  
│　　├── `__`init`__`.py  
  

方案：

```python
import importlib
# 导入
params = importlib.import_module("b.c.c") 
# 对象中取出需要的对象
params.args   # 取出变量
params.C      # 取出类C
```

