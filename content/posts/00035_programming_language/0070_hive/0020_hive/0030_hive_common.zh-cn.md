---
title: 常用操作
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 常用操作
    identifier: posts-hive-hive-common
    parent: posts-hive-hive
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Hive", "common"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
aliases:
  - /zh-cn/posts/00035_programming_language/0070_hive/0020_hive/0030_hive_common/
---

## 常用功能

### 1、获取当前时间
```sql
from_unixtime(unix_timestamp(),'yyyy-MM-dd HH:mm:ss')
```

### 2、多行合并为一行

```sql
select gid, concat_ws(',',collect_set(url)) 
from table
group by gid;
```

### 3、一行拆多列

```sql
-- 1列
select gid, url 
from table 
lateral view explode(split(url_list, ','))t as url

-- 2列
select gid, url, url_flag
from table
lateral view posexplode(split(url_list, ','))t1 as url_index, url
lateral view posexplode(split(url_flag_list, ','))t2 as url_flag_index, url_flag
where url_index = url_flag_index;

```

### 4、添加行号
如果单纯想添加一个自增的行号，没有顺序要去的话，over（）里面可以空着
```sql
Select row_number() over() as rownum
From table
```

### 5、窗函数

```sql
SELECT
	RANK ()       over (PARTITION BY SUBJECT ORDER BY score DESC) AS RANK,
	DENSE_RANK () over (PARTITION BY SUBJECT ORDER BY score DESC) AS DENSE_RANK,
	ROW_NUMBER () over (PARTITION BY SUBJECT ORDER BY score DESC) AS ROW_NUMBER
FROM table;
```
<p align="center"><img src="/datasets/posts/hive/hive_window.png" width=70% height=70%></p>


### 6、条件语句
【nvl】
```sql
nvl(exp1, exp2)  -- 如果第一个不为空，就返回第一个；如果为空返回第二个
COALESCE(表达式1，表达式2，表达式3, ....) -- 适合多个参数
```

【IF】
```sql
if(判断项，表达式1， 表达式2)
```

【case】
```sql
case 表达式
when 值1 then 表达式1
when 值2 then 表达式2
else 表达式3
end
```

### 7、调用py脚本

```sql

add archive ***/anaconda3.tar.gz
add file /path_of_python_file/py_file.py;

-- 方法一：
-- transform的参数col1,col2,col3,...作为python脚本的输入，out1,out2,out3,...作为输出字段。
select transform(col1,col2,col3,...) 
using 'python py_file.py' 
as (out1,out2,out3,...)
from table;

-- 方法二：
reduce co11,col2,col3,... 
using '/home/sharelib/python py_file.py'
as out1,out2,out3,...
from table;
```

### 8、字符串是否包含子串
regexp

```sql
'abcd'  regexp 'ab'
```

### 9、正则-删掉一些特殊字符
```sql
regexp_replace(title, '(\n)|(\r)|(\t)', ' ') as title
```

### 10、窗口中排序
sort_array

```sql
select
    province,
    concat_ws(',', collect_list(city)) as 行转列,
    concat_ws(
        ',',
        sort_array(
            collect_list(
                concat_ws(':', lpad(row_number_score, 5, '0'), city)
            )
        )
    ) as 中间值,
    regexp_replace(
        concat_ws(
            ',',
            sort_array(
                collect_list(
                    concat_ws(':', lpad(row_number_score, 5, '0'), city)
                )
            )
        ),
        '\\d+:',
        ''
    ) as 最终结果
from (
    SELECT
        province
        , city
        , score
        , row_number() over(partition by province order by score desc) as row_number_score
    FROM temp
    having row_number_score <= 5
)
group by province;

```
求出的结果如下所示：
|province|行转列|中间值|最终结果|
|:--|:--|:--|:--|
|广东|广州,佛山,东莞,中山|00001:广州,00002:佛山,00003:东莞,00004:中山|广州,佛山,东莞,中山|
|湖南|长沙,株洲,湘潭,娄底,邵阳|00001:长沙,00002:株洲,00003:湘潭,00004:娄底,00005:邵阳|长沙,株洲,湘潭,娄底,邵阳|


### 11、提取表情

```sql
SELECT regexp_replace('风蛋包饭🌈实在太好吃啦😋 昨天复刻', '([\\p{P}+~$`^=|<>～｀＄＾＋＝｜＜＞￥×\\u4E00-\\u9FA5a-zA-Z0-9]+)', '') as subtitle
```



