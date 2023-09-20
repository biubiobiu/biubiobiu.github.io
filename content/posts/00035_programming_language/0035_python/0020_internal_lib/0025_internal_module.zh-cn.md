---
title: "内置模块"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: 内置模块
    identifier: python-internal-modules
    parent: python-internal
    weight: 25
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","内置模块"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、os

|||
|:--|:--|
|os.path.basename()||
|os.path.dirname()||
|os.path.join()||
|os.path.exists()||
|os.path.isfile()||
|os.path.isdir()||
|os.path.abspath(`__file__`)|获取当前执行文件的绝对路径|
|os.listdir()|遍历该目录下的文件，返回文件名列表|
|os.walk()|遍历该目录，返回的是一个三元组(root, dirs, files)<br>root: 指的是当前正在遍历的文件夹的地址<br>dirs: 是一个list，内容是该文件夹中所有的目录的名字，不包括子目录<br>files：内容是该文件夹中所有的文件，不包括子目录|
|os.makedirs()|创建文件夹|
|os.remove()|删除文件夹|
|os.environ()|获取环境变量，比如：os.environ('变量名', '默认值')|



## 二、sys

|||
|:--|:--|
|sys.path|搜索路径|
|sys.platform|获取当前系统平台|
|sys.argv|实现从程序外部向程序传递参数|
|sys.exit([arg])|程序中间的退出，arg=0为正常退出，例如：sys.exit(0)|
|sys.getdefaultencoding()|获取当前系统编码，一般为ascii|
|sys.setdefaultencoding()|设置系统默认编码，比如：sys.setdefaultencoding('utf8')|
|sys.getfilesystemencoding()|获取文件系统使用编码方式：<br>windoes: 'mbcs'<br>mac: 'utf-8'|
|sys.stdin()||
|sys.stdout()||
|sys.stderr()||


## 三、内置函数

### 1、字符判断

|字符串检测方法||
|:--|:--|
|<font color=#f00000>isalnum()</font>|检测字符串是否由字母和数字组成|
|<font color=#f00000>isalpha()</font>|检测字符串是否只由字母组成。|
|<font color=#f00000>isascii()</font>|检测字符串是否都是ASCII编码的字符|
|<font color=#f00000>isdigit()</font>|检测字符串是否只由数字组成|
|<font color=#f00000>islower()</font>|检测字符串是否由小写字母组成|
|<font color=#f00000>isupper()</font>|检测字符串是否由大写字母组成|
|<font color=#f00000>isdecimal()</font>|检查字符串是否只包含十进制字符|
|<font color=#f00000>isidentifier()</font>|判断字符串是否是有效的Python标识符|
|<font color=#f00000>isnumeric()</font>|检测字符串是否只由数字组成|
|<font color=#f00000>isspace()</font>|检测字符串是否只由空白字符组成|
|<font color=#f00000>istitle()</font>|检测字符串中所有的单词拼写首字母是否为大写，且其他字母为小写|


