---
title: 创建库/表
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 创建库/表
    identifier: posts-hive-hive-build
    parent: posts-hive-hive
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Hive"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数据库操作

### 1、创建数据库

|功能|操作|
|:--|:--|
|查看数据库|show databases; <br> 使用like关键字模糊匹配，显示包含db_前缀的数据库名称    show databases like 'db_*';|
|使用数据库 |create database dbname;|
|创建数据库|create database dbname; <br> 通过location指定数据库路径   create database dbname location 'path路径'; <br> 给数据库添加描述信息   create database dbname comment 'dbname描述信息';|
|删除数据库|删除数据库，这种删除，需要将数据库中的表全部删除，才能删除数据库 <br> drop database dbname; <br> 或者 <br> drop database if exists dbname; <br> cascade 强制删除   drop database dbname cascade;|
|查看数据库的详细描述|desc database dbname; <br> destribe database dbname;|


## 二、表操作

|功能|操作|
|:--|:--|
|创建表|CREATE TABLE 表名|
|删除表|drop table if exists 表名|
|添加列|alter table 表名 add columns (列名 string comment '解释')|
|修改字段类型|alter table 表名 change column 原字段名称 现字段名称 数据类型|


