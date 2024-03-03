---
title: 数据类型
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 数据类型
    identifier: posts-hive-hive-datatype
    parent: posts-hive-hive
    weight: 40
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Hive", "regular"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数组array

【语法】： 
```sql
array(val1, val2, val3, ...)
```

【建表】
```sql
create table temp.array_demo(
    meiti array<string> comment ''
)
row format delimited 
fields terminated by '\t'  -- (字段与字段之间的分隔符)
lines terminated by '\n'  ()
collection items terminated by ','  -- (必须使用, 一个字段中各个item的分割符)
lacation 'hdfs 路径'

```

【查询方法和函数】

```sql
-- 在字段类型为array中查找是否包含某元素
array_contains()  

-- 数组名[index]：查询

```


## 二、map

|函数|说明|
|:--|:--|
|size(Map)|map的长度|
|map_keys(Map)|map中的所有key，返回array|
|map_values(Map)|map 中所有的value，返回array|
|【构建Map】|组装数据：to_json, str_to_map <br> str_to_map(concat_ws(",", collect_set(concat_ws(':', date_key, price_value))))|
|【构建Map】|聚合：combine；聚合去重：combine_unique <br> combine(map1, map2, map3, ...)|
|【构建Map】|hive 自带方法 <br> map("key1", value1, "key2", value2) <br> named_struct("cnt", 100, "uds", array(2, 3, 4))|


