---
title: "Toha的配置"
date: 2020-06-08T06:00:20+06:00
menu:
  sidebar:
    name: Toha的配置
    identifier: toha-config-github
    parent: toha-tutorial
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Toha","配置"]
categories: ["Basic"]
---
## 一、启动

模板项目: <a href="https://github.com/hugo-toha" target="blank">github</a>

```shell
# --force 即使本文件夹不为空，也会强制创建站点
hugo new site myblog -f=yaml --force
# 初始化本地仓库，因为部署时要把该文件的内容push到远端仓库
git init 
# 添加toha主题
git submodule add https://github.com/hugo-toha/toha.git themes/toha
# 在本地启动站点，浏览器中打开: http://localhost:1313
hugo server -t toha -w

```

<a href="https://hugo-toha.github.io/" target="blank">Demo样例</a>

<a href="https://gohugo.io/content-management/multilingual/" target="blank">Hugo文档</a>

<a href="https://github.com/hugo-toha" target="blank">Github项目</a>

## 二、配置

config.yaml: <a href="https://github.com/hugo-toha/hugo-toha.github.io/blob/source/config.yaml" target="blank">配置样例</a>

这个主题的大部分内容是由`data`目录中的一些 YAML 文件驱动的。 在本节中，我们将添加一些示例数据。 由于我们正在构建一个多语言站点，因此我们会将每种语言的数据保存在各自的`语言环境文件夹`中。首先，在`data`目录中创建 en 文件夹(英语环境)/zh-cn(汉语环境)。 我们将在这里添加英语语境数据。

### 1、主页配置

在目的环境文件夹中创建site.yaml

英语环境：`/data/en/site.yaml` 汉语环境：`/data/zh-cn/site.yaml`

```yaml
# Copyright Notice
copyright: © 2021 Copyright.

# A disclaimer notice for the footer. Make sure you have set "params.footer.disclaimer.enable: true" in your `config.yaml` file.
disclaimer: "这个主题是MIT许可的"

# Meta description for your site.  This will help the search engines to find your site.
description: 机器学习、深度学习 探索者.

# 指定要在顶部导航栏中显示的自定义菜单列表。它们将通过分隔线与主菜单分开。
customMenus:
- name: 文档
  url: https://toha-guides.netlify.app/posts/

# Specify OpenGraph Headers
openGraph:
  title: biubiobiu's Blog
  type: website
  description: biubiobiu的简历和私人博客.
  image: images/author/john.png
  url: https://***.github.io
```



### 2、作者信息配置

在目的语言环境路径中创建：`author.yaml`文件

英语环境: `/data/en/author.yaml`

汉语环境: `/data/zh-cn/author.yaml`

### 3、区域块设置

在目的语言环境路径中创建：`sections`文件夹

英语环境：`data/en/sections/`

汉语环境：`data/zh-cn/sections/`



## 三、部署

It's coming soon ... 
