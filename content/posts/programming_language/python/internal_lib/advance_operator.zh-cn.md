---
title: "进阶操作"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 进阶操作
    identifier: python-advance_operator
    parent: python-internal
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
---

## 一、环境变量

### 1、`临时环境变量`
|操作|说明|功能|
|:---|:---|:---|
|`os.environ['WORKON_HOME']="变量"`|设置环境变量||
|`os.environ.get('WORKON_HOME')`|获取环境变量-方法1||
|`os.getenv('path')`|获取环境变量-方法2-推荐||
|`del os.environ['WORKON_HOME']`|删除环境变量||
||||
|`os.environ['HOMEPATH']`|当前用户主目录||
|`os.environ['TEMP']`|临时目录路径||
|`os.environ['PATHEXT']`|可以执行文件||
|`os.environ['SYSTEMROOT']`|系统主目录||
|`os.environ['LOGONSERVER']`|机器名||
|`os.environ['PROMPT']`|设置提示符||

### 2、`永久环境变量`

|操作|说明|功能|
|:---|:---|:---|
|path = r"路径"<br>command = r"setx WORK1 %s /m"%path<br>os.system()|/m 表示系统变量，不加/m表示用户变量||


### 3、`内部变量`

|操作|说明|功能|
|:---|:---|:---|
|`__doc__`|获取文件的注释||
|`__file__`|获取当前文件的路径||
|`__name__`|获取导入文件的路径加文件名称。当前文件，其值为__main__||
|`__package__`|获取导入文件的路径。当前文件，其值为 None||
|`__cached__`|||
|`__builtins__`|内置函数在这里||

实例：获取该执行文件的绝对路径：`os.path.dirname(os.path.abspath(__file__))`


## 二、yield

带有yield函数在python中被称为generator。以菲波那切数列为例，介绍一下，yield的功能：
1. 输出菲波那切数列list<br>
缺点：返回list，运行时占用的内存随着参数max的增大而增大，如果要控制内存，最好不要用list来存储中间结果，而是通过iterable对象来迭代。
```python
def fab(max):
    n, a, b = 0, 0, 1
    L = []
    while n < max:
        L.append(b)
        a, b = b, a+b
        n = n + 1
    return L
```

2. iterable 的方法实现：通过next()函数不断返回数列的下一个数，内存占用始终未常数。<br>
缺点：不够简洁
```python
class Fab(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1
    def __iter__(self):
        return self
    def next(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a+self.b
            self.n = self.n+1
            return r
        raise StopIteration()

# 调用
for n in Fab(5):
    print(n)
```

3. 使用yield：yield把一个函数变成一个generator，调用fab()函数时不会执行该函数，而是返回一个iterable对象。在for循环执行时，每次循环都会执行fab函数内部的代码。
```python
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b 
        a, b = b, a+b
        n = n+1
# 调用
for n in fab(5):
    print(n)
```

## 三、闭包

```python
# 定义
def line_conf(a, b):
    def line(x):
        return a*x+b
    return line

# 定义两条直线
line_a = line_conf(2,1) # y=2x+1
line_b = line_conf(3,2) # y=3x+2

print(line_conf().__closure__)  # 闭包函数的属性
```

1. 闭包函数的必要条件
    - 闭包函数(例如：line_conf())，必须返回一个函数对象
    - 闭包函数返回的函数对象(例如：line())，必须引用外部变量(一般不能是全局变量)，而返回的那个函数对象(例如：line())内部不一定要return

2. 作用域分析
    - 函数的作用域是由def关键词界定的，函数内的代码访问变量的方式是：从其所在层级由内向外寻找
    - 函数属性：闭包函数将函数的唯一实例保存在它内部的__closure__属性中，在再次创建函数实例时，闭包检查函数实例已存在自己的属性中，不会再让它创建新的实例，而是将现有的实例返回。


## 四、装饰器

1. 实例
```python
# 定义
def a_new_decorator(a_func):
    def wrapTheFunction():
        print('I am doing some boring work before executing a_func()')

        a_func()

        print('I am doing some boring work after executing a_func()')
    return wrapTheFunction

def a_function_requiring_decoration():
    print('I am the function which needs some decoration to remove my foul smell.')

# 调用
a_function_requiring_decoration()
# 结果：I am the function which needs some decoration to remove my foul smell.

a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
a_function_requiring_decoration()
# 结果：
# I am doing some boring work before executing a_func()
# I am the function which needs some decoration to remove my foul smell.
# I am doing some boring work after executing a_func()

```

2. 用@简化代码<br>
缺点：print(a_function_requiring_decoration.__name__) 返回的是装饰器: wrapTheFunction
```python
# 定义
def a_new_decorator(a_func):
    def wrapTheFunction():
        print('I am doing some boring work before executing a_func()')

        a_func()

        print('I am doing some boring work after executing a_func()')
    return wrapTheFunction

@a_new_decorator
def a_function_requiring_decoration():
    print('I am the function which needs some decoration to remove my foul smell.')

# 调用
a_function_requiring_decoration()
# 结果：
# I am doing some boring work before executing a_func()
# I am the function which needs some decoration to remove my foul smell.
# I am doing some boring work after executing a_func()

```


3. @蓝本 <br>
可以利用@wraps接受一个函数进行装饰，并加入复制函数名称、注释文档、参数列表等功能。

```python
# 定义
from functools import wraps
def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print('I am doing some boring work before executing a_func()')

        a_func()

        print('I am doing some boring work after executing a_func()')
    return wrapTheFunction

@a_new_decorator
def a_function_requiring_decoration():
    print('I am the function which needs some decoration to remove my foul smell.')

# 调用
a_function_requiring_decoration()
# 结果：
# I am doing some boring work before executing a_func()
# I am the function which needs some decoration to remove my foul smell.
# I am doing some boring work after executing a_func()

print(a_function_requiring_decoration.__name__)
# 结果：a_function_requiring_decoration

```

## 五、内置函数

1. `eval('字符串')`：把字符串作为语句执行
作用：解析并执行字符串，并将返回结果输出。eval()函数将去掉字符串的两个引号，将其解释为一个变量。
    -  1）单引号，双引号，eval()函数都将其解释为int类型；eval('100')，输出的是int类型。
    -  2）三引号则解释为str类型。eval('"hello"')，输出的是字符串

2. `input()` : 键盘输入
作用：接收键盘的输入，返回的是字符串类型。
input和eval函数结合使用：
    - 1）从键盘输入，接收一个字符串类型： a = input('请输入一个字符串：')
    - 2）从键盘输入，接收一个整型： a = eval(input('请输入一个数字：')) 

3. `lambda`  匿名函数  
格式：lambda[arg1[,arg2,...,argN]] : 表达式 <br>
例如：test = lambda x, y: x+y 

4. `sorted`  排序
b=sorted(a.items(), key=lambda item:item[0], reverse = True)
    - a.items()  表示可迭代的tuple列表
    - key=lambda item:item[0]：按照key值排序; lambda x:x[0] 
    - reverse = True：降序排序

## 六、内置模块-os

|操作|解释|
|:--|:--|
|os.path.basename()||
|os.path.dirname()||
|os.path.join()||
|os.path.exists()|判断该路径是否存在|
|os.path.isfile()|判断是不是文件|
|os.path.isdir()|判断是不是目录|
|os.path.abspath()|获取绝对路径<br>例如：os.path.abspath(__file__)获取当前文件的绝对路径|
|os.listdir()|遍历该目录下的文件，返回文件名列表|
|os.walk()|遍历目录，返回一个三元组(root,dirs,files)<br> root: 指的是当前正在遍历的文件夹本身的目录<br>  dirs: 是一个list，内容是该文件夹中所有的目录的名字(不包含子目录)<br>  files: 是一个list，内容是该文件夹中所有的文件(不包括子目录)|
|os.makedirs|创建一个目录|
|os.remove|删除一个目录|
|os.environ|环境变量<br> 例如：获取环境变量：os.environ.get('环境变量名', '默认值')|

