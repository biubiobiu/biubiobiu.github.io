---
title: "py-env"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: py-env
    identifier: python-env-pip
    parent: python-env
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","pip"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、anaconda环境

<a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/" target="bland">清华镜像源</a>

1. 可以通过从页面上下载，直接安装
2. 可以是命令
    * wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2022.10-MacOSX-x86_64.sh
    * sh Anaconda3-2022.10-MacOSX-x86_64.sh
    * 配置环境变量：export PATH=~/anaconda3/bin:$PATH


|操作|说明|
|:--|:--|
|conda config `--show`|查看配置|
|conda config <br>`--add` channels 网址|添加源<br>conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/<br>conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/<br>conda config --set show_channel_urls yes|
|conda info -e|查看conda的虚拟环境|
|conda list|查看该环境下，已经安装的包-版本|
|conda search 包名|查看安装包，是否可通过conda安装|
|<kbd>安装</kbd>||
|conda install <br>-n 环境名 包名<br>-c 网址|在指定虚拟环境下安装，-c：全名是--channel，指定来源，例如：<br>conda install -c https://conda.anaconda.ort/pandas   bottleneck|
|conda remove <br> -n 环境名 包名|从指定虚拟环境下卸载|
|<kbd>虚拟环境</kbd>||
|conda create -n 环境名 python=3.8|创建一个虚拟环境，指定安装python的版本为3.8。-n全名是`--name`|
|conda remove -n 环境名 `--all`|删除一个虚拟环境|
|source activate 环境名|切换到一个虚拟环境|
|source deactivate|退出该环境|
|<font color=#a020f0>根据yml文件，创建环境、安装</font><br><br>conda env create <br>-f environment.yml|根据yml文件内容，创建环境、安装各种包<br>yml文件内容，例如：<br>name: py38<br>dependencies:<br>- python=3.8<br>- pip:<br>　- mxnet==1.5.0<br>　- pandas==0.23.4<br>　- matplotlib==2.2.2<br>|
|conda env export > environment.yml|导出虚拟环境，生成.yml文件。|

## 二、pypi

|国内pypi源|链接|
|:--|:--|
|清华大学|`https://pypi.tuna.tsinghua.edu.cn/simple/`|
|中国科学技术大学|`http://pypi.mirrors.ustc.edu.cn/simple/`|
|阿里云|`http://mirrors.aliyun.com/pypi/simple/`|
|豆瓣|`https://pypi.douban.com/simple/`|


|pip 操作|说明|
|:--|:--|
|<kbd>pip配置</kbd>||
|pip config set global.index-url 网址|conda和pip默认国外站点下载，我们可以配置成国内镜像来加速下载|
|pip config unset global.index-url|取消pypi镜像配置|
|<kbd>安装</kbd>||
|pip install -i 网址 <包名>|安装，-i全名是`--index`，指定下载源|
|pip install <路径>/<包名>|安装本地的包|
|pip install <包名> `--upgrade`|升级包|
|pip uninstall <包名>|卸载|
|pip show -f <包名>|显示包所在目录|
|pip search <关键词>|搜索包|


## 三、环境变量

新安装的工具包，执行文件的路径需要添加到环境变量中，系统才能访问到。可以使用<font color=#a020f0>export</font>命令添加。
|export 操作|说明|
|:--|:--|
|export [-fnp] <br>[变量名]=[变量设置值]|-f：代表变量名称中为函数名称<br>-n：删除指定的变量，变量实际上未被删除，只是不会输出到后续指令的执行环境中<br>-p：列出所有的shell赋予程序的环境变量|
||例如：mongodb 包|
|<kbd>方法一：</kbd>|<font color=#a020f0>export PATH=/usr/local/mongodb/bin:$PATH</font><br>生效方式：立即生效<br>有效期：临时改变，只能在当前的终端窗口中生效，当前窗口关闭后就会回复原有的path配置<br>用户局限：仅对当前用户|
|<kbd>方法二：</kbd>|<font color=#a020f0>修改.bashrc文件，或者.zshrc文件。</font>这个要看用的是那个<br>在文件中添加：export PATH=/usr/local/mongodb/bin:$PATH<br>生效方式：执行 `source .zshrc` 或者 关闭当前终端，重新打开一个<br>有效期：永久<br>用户局限：仅对当前用户|
|<kbd>方法三：</kbd>|<font color=#a020f0>修改/etc/profile文件</font><br>在文件中添加：export PATH=/usr/local/mongodb/bin:$PATH<br>生效方式： `系统重启`<br>有效期：永久<br>用户局限：所有用户|
