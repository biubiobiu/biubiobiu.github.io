---
title: "requests"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: requests
    identifier: python-sdk-requests
    parent: python-sdk
    weight: 70
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","request"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、request模块

requests模块：在python内置模块上进行了高度的封装，使得requests更方便。<br>
url: uniform resource locator，统一资源定位符：互联网上标准资源的地址。<br>
格式：
1. 模式/协议，比如：https、http
2. 服务器名称(或者IP地址)，比如：api.github.com
3. 路径和文件名，比如：events

||||
|:--|:--|:--|
|requests.get(url)||get请求 --- 不带参数|
|requests.get(url, params={"参数1":"值1"})||get请求 --- 带参数|
|requests.get(url, headers=header, cookie=cookie)|header = {"content-type": "application/json","user-agent": ""}|定制headers|
|requests.get(url, proxies=proxies)|proxies = {"http": "ip1", "https": "ip2"}|代理|
||||
|requests.post(url, data=json.dumps({"":""}))||post请求|
|requests.post(url, headers=header, cookie=cookie)|cookie = {"cookie":"cookie_info"}|定制cookie|
||||
|<kbd>Session()</kbd>|会话对象，能够跨请求保持某些参数。|会话|
|s = requests.Session()|header = {"user-agent":"",...}<br>s.header.update(header)<br>s.auth = {"auth", "password"}<br>response = s.get(url) 或者  s.port(url)||
|from requests.auth import <kbd>HTTPBasicAuth</kbd>|另一种非常流行的http身份认证形式：摘要式身份认证|身份认证|
|response = requests.get(url, auth=HTTPBasicAuth("user","password"))|requests.get(url, HTTPDigestAuth("user","password"))||

<br>

requests对象的get和post方法都会返回一个Response对象。这个对象里面存的是服务器返回的所有信息：
1. 响应头、
2. 响应状态码；
3. 其中返回的网页部分会存在content和text两个对象中。
   1. content：字节码  （有中文时，.content.decode('utf-8')）
   2. text   ：字符串，beautifulsoup根据猜测的编码方式将content内容编码成字符串。

get 与post的不同：
1. get方式，通过url提交数据；post方式，数据放置在header内
2. get方式，提交的数据最多只有1024Byte，post没有限制
3. get方式，从服务器取数据，url中会有少量的信息传送给服务器，用于说明要取什么样的数据； post方式是被设计用来向上放东西的，向服务器传送的是HTTP请求的内容，

|response对象的操作|||
|:--|:--|:--|
|response.url|||
|response.encoding|||
|response.text|以encoding解析返回内容。根据响应头部的编码方式进行解码|字符串  内容|
|response.content|以字节(二进制)返回。会自动解析gzip和deflate压缩|二进制 内容|
|response.json()|模块内置的json解码器。以json形式返回，前提是返回的内容确实是json格式，否则会报错。|json格式  内容|
|response.headers|服务器响应头部，字典形式。||
|response.request.headers|返回发动到服务器的头信息||
|response.status_code|响应状态码||
|response.raise_for_status()|失败请求抛出的异常||
|response.cookies|返回响应中包含的cookie||
|response.history|||
|response.elapsed|返回timedelta, 响应所用的时间|响应时间|

