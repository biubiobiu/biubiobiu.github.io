---
title: 正则匹配
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 正则匹配
    identifier: posts-hive-hive-regular
    parent: posts-hive-hive
    weight: 50
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Hive", "regular"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
aliases:
  - /zh-cn/posts/00035_programming_language/0070_hive/0020_hive/0050_hive_regular/
---

## 一、常用的两个函数

### 1、regexp_extract

【语法】：regexp_extract(string subject, string pattern, int index) <br>
【返回】：string <br>
【说明】：将字符串subject按照pattern正则表达式的规则拆分，返回index指定的字符 <br>
　　　　pattern中 '()()()()'   <br>
　　　　index 表示输出第index个括号的正则表达式 <br>
　　　　　　0-表示与之匹配的整个字符串 <br>
　　　　　　1-表示第一个括号里面的 <br>
　　　　　　2-表示第二个括号里面的<br>
　　　　例如：<br>
　　　　select regexp_extract('x=a3&x=18abc&x=2&y=3&x=4','x=([0-9]+)([a-z]+)',0);  --- x=18abc<br>
　　　　select regexp_extract('x=a3&x=18abc&x=2&y=3&x=4','x=([0-9]+)([a-z]+)',2);  --- abc<br>


### 2、regexp_replace

【语法】：regexp_replace(string A, string B, string C) <br>
【返回】：string <br>
【说明】：将字符串A中的符号java正则表达式B的部分提换为C，<br>


## 二、正则匹配项

|操作|解释说明|功能|
|:--|:--|:--|
|Java正则|"." 任意单个字符 <br> "*" 匹配前面的字符0次或多次 <br> "+" 匹配前面的字符1次或多次[] <br> "?" 匹配前面的字符0次或1次 <br> "\d" 等于 [0-9]，使用的时候写成'\d' <br> "\D" 等于 [^0-9]，使用的时候写成'\D'  非数字符 <br> '\\W*' 匹配汉字 <br> '\\w*'  匹配字母、下划线、数字 ||
||regexp_replace('', '([^\\u4E00-\\u9FA5a-zA-Z0-9]+)', '')|匹配：中文、英文、数字|
|\s+|\s+: 匹配多个空格 <br>例如：this\s+is\s+test <br>匹配单词this后面的\s+可以匹配多个空格；之后匹配is；再之后\s+匹配多个空格再加上text字符串||
|^|定义了以什么开始 <br> 例如：^\d+(\.\d+)?||
|$|定义了以什么结尾||
|+|||
||||
||||
||||
||||
||||


