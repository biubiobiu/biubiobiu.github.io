---
title: "logging"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: logging
    identifier: python-sdk-logging
    parent: python-sdk
    weight: 80
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","logging"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、logging模块

logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等；相比print，具备如下优点：
1. 可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息；
2. print将所有信息都输出到标准输出中，严重影响开发者从标准输出中查看其它数据；logging则可以由开发者决定将信息输出到什么地方，以及怎么输出；


> logging模块与log4j的机制是一样的，只是具体的实现细节不同。模块提供logger，handler，filter，formatter。
> 1. logger：提供日志接口，供应用代码使用。logger最长用的操作有两类：配置和发送日志消息。可以通过logging.getLogger(name)获取logger对象，如果不指定name则返回root对象，多次使用相同的name调用getLogger方法返回同一个logger对象。
> 2. handler：将日志记录（log record）发送到合适的目的地（destination），比如文件，socket等。一个logger对象可以通过addHandler方法添加到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示。
> 3. filter：提供一种优雅的方式决定一个日志记录是否发送到handler。
> 4. formatter：指定日志记录输出的具体格式。formatter的构造方法需要两个参数：消息的格式字符串和日期字符串，这两个参数都是可选的。
> 
> 与log4j类似，logger，handler和日志消息的调用可以有具体的日志级别（Level），只有在日志消息的级别大于logger和handler的级别。


```python
import logging 

#
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

# 创建一个FileHandler
handler = logging.FileHandler('log.txt')

# 设置等级: DEBUG < INFO < WARNING < ERROR < CRITICAL，而日志的信息量是依次减少的
handler.setLevel(logging.INFO)

# 设置输出消息的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 添加到logger中
logger.addHandler(handler)

# 写入消息
logger.info("Hello")
```

## 二、消息格式
|输出消息的格式|解释|
|:--|:--|
|%(levelno)s|打印日志级别的数值|
|`%(levelname)s`|打印日志级别的名称|
|%(pathname)s|打印当前执行程序的路径，其实就是sys.argv[0]|
|%(filename)s|打印当前执行程序名|
|%(funcName)s|打印日志的当前函数|
|%(lineno)d|打印日志的当前行号|
|`%(asctime)s`|打印日志的时间|
|%(thread)d|打印线程ID|
|%(threadName)s|打印线程名称|
|%(process)d|打印进程id|
|`%(message)s`|打印日志信息|

