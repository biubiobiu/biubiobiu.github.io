---
title: "并行操作"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 并行操作
    identifier: python-sdk-multprocess
    parent: python-sdk
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
---

## 一、线程与进程


||进程|线程|
|:---|:---|:---|
||`进程`：是一个应用程序在处理机上的一次执行过程，是具有一定独立功能的程序在某数据集上的一次运行，是一个`动态`的概念。进程是系统进行资源分配和调度的独立单位。|`线程`：是进程中的一个实体，是CPU调度和分派的基本单位，线程自己基本上不拥有系统资源，它与同属于一个进程内的其他线程共享进程的全部资源。|
|地址空间|进程有自己独立的地址空间|进程中至少有一个线程，它们共享进程的地址空间|
|资源|进程是资源分配和拥有的单位|进程内的多个线程共享进程的资源|
|调度||`线程`是进程内的一个执行单元，也是进程内的可调度实体，也是处理器调度的基本单位|


## 二、多线程

### 1、`threading模块`

python主要是通过`thread`和`threading`这两个模块来实现多线程，`thread`模块是比较底层的模块，`threading`模块是对thread做了一些封装，使用更方便。但是由于GIL的存在，无法使用threading充分利用CPU资源，如果想充分发挥多核CPU的计算能力，需要使用`multiprocessing`模块<br>

python 3.x 已经摒弃了python 2.x中采用函数式thread模块来产生线程的方式。而是通过threading模块创建新的线程：
1. 通过`threading.Thread(Target=可执行方法)`<br>
    ```python
    import threading
    pro_list = []
    mult_image_label_list = []
    for index, img_list in enumerate(mult_image_label_list):
        # 创建线程
        t1 = threading.Thread(target=函数名, args=(index, img_list))
        pro_list.append(t1)
    
    for thread in pro_list:
        # 将线程设置为保护线程，否则会被无限挂起。
        thread.setDaemon(True)
        thread.start()
    
    # 该位置---子线程与父线程同时执行，父线程执行完后，同时结束子线程的执行。
    # 如果不添加join()语句，父线程结束后，子线程就会结束。
    # 如果需要在子线程执行完后，父线程才结束，需要添加join()，让父进程一直处于阻塞状态，直到所有子线程执行完毕。
    
    for thread in pro_list:
      # 在子线程结束前，父线程一直处于阻塞状态。让子线程执行完，才执行父线程，添加join()。
      thread.join()
    
    ```

2. 继承`threading.Thread`定义子类，并重写`run()`方法和`init()`<br>实例化后调用start()方法启动新线程，即：它调用了线程的run()方法。

    ```python
    import threading
    import time
    
    class myThread(threading.Thread):
      def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    
      def run(self):
        print("Starting " + self.name)
        print_time(self.name, self.counter, 5)
        print("Exiting " + self.name)
    
    
    def print_time(threadName, delay, counter):
       while counter:
          time.sleep(delay)
          print("%s process at: %s" % (threadName, time.ctime(time.time())))
          counter -= 1


    # 创建新线程
    thread1 = myThread(1, "Thread-1", 1)
    thread2 = myThread(2, "Thread-2", 2)
    
    # 开启线程
    thread1.start()
    thread2.start()
    
    # 等待线程结束
    thread1.join()
    thread2.join()
    
    print("Exiting Main Thread")
    
    ```
    上例中thread1和thread2执行顺序是乱序的，如果要使其有序，需要进行线程同步。<br>
    如果多个线程共同对某个数据操作，可能会出现不可预料的结果，为了保证数据的正确性，需要对多个线程进行同步。`threading.Lock()`有acquire方法(进行加锁)和release方法(进行解锁)，对于需要每次只允许一个线程操作的数据，可以将其操作放在acquire和release方法之间。`线程同步的方式：锁机制、同步队列`
    1. 锁机制
    ```python
    class myThread(threading.Thread):
        def __init__(self, threadID, name, counter, lock):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.counter = counter
            self.lock = lock
    
        def run(self):
            print("Starting " + self.name)
            # 加锁
            self.lock.acquire()
            print_time(self.name, self.counter, 5)
            # 解锁
            self.lock.release()
            print("Exiting " + self.name)
    
    
    def print_time(threadName, delay, counter):
        while counter:
            time.sleep(delay)
            print("%s process at: %s" % (threadName, time.ctime(time.time())))
            counter -= 1
    lock = threading.Lock()

    # 创建新线程
    thread1 = myThread(1, "Thread-1", 1, lock)
    thread2 = myThread(2, "Thread-2", 2, lock)

    # 开启线程
    thread1.start()
    thread2.start()

    # 等待线程结束
    thread1.join()
    thread2.join()

    print("Exiting Main Thread")
    ```

    2. 线程同步队列queue<br>
    python 2.x 提供的Queue，python3.x中提供的是queue。其中queue模块找那个提供了同步的、线程安全队列类，包括：FIFO(先入先出队列)、LIFO(后入先出队列)、PriorityQueue(优先级别队列)。这些队列都实现了锁原语，能够在多线程中直接使用。<br>
    可以使用队列来实现线程间的同步。
    |queue常用方法||
    |:---|:---|
    |queue.qsize()|返回队列的大小|
    |queue.empty()|如果队列为空，返回True，否则返回False|
    |queue.full()|如果队列满了，返回True，否则返回False|
    |queue.get()|获取队列|
    |queue.get_nowait()|相当于Queue.get(False)|
    |queue.put()|写入队列|
    |queue.put_nowait(item)|相当于Queue.put(item, False)|
    |queue.task_done()|在完成一项工作之后，向任务已经完成的队列发送一个信号|
    |queue.join()|实际上意味着等到队列为空，再执行别的操作|
    
    ```python
    #!/usr/bin/python3

    import queue
    import threading
    import time

    exitFlag = 0

    class myThread (threading.Thread):
        def __init__(self, threadID, name, q):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.q = q
        def run(self):
            print ("开启线程：" + self.name)
            process_data(self.name, self.q)
            print ("退出线程：" + self.name)

    def process_data(threadName, q):
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                data = q.get()
                queueLock.release()
                print ("%s processing %s" % (threadName, data))
            else:
                queueLock.release()
            time.sleep(1)

    threadList = ["Thread-1", "Thread-2", "Thread-3"]
    nameList = ["One", "Two", "Three", "Four", "Five"]
    queueLock = threading.Lock()
    workQueue = queue.Queue(10)
    threads = []
    threadID = 1

    # 创建新线程
    for tName in threadList:
        thread = myThread(threadID, tName, workQueue)
        thread.start()
        threads.append(thread)
        threadID += 1

    # 填充队列
    queueLock.acquire()
    for word in nameList:
        workQueue.put(word)
    queueLock.release()

    # 等待队列清空
    while not workQueue.empty():
        pass

    # 通知线程是时候退出
    exitFlag = 1

    # 等待所有线程完成
    for t in threads:
        t.join()
    print ("退出主线程")
    ```

### 2、`ThreadPoolExecutor线程池`

> `传统多线程问题`：一个线程的运行时间可以分为3部分：线程的启动时间、线程体的运行时间、线程的销毁时间。`如果线程不能被重用，这就意味着每次创建都需要经过启动、运行、销毁这3个过程。`这必然会增加系统响应的时间，降低效率。另外一种高效的解决方法——线程池。<br>
> `线程池`：把任务放进队列中，然后开N个线程，每个线程都取队列中取一个任务，执行完了之后告诉系统我执行完了，然后接着从队列中取下一个任务，直至队列中所有任务取空，退出线程。由于线程预先被穿件并放入线程池中，同时处理完当前任务之后并不销毁而是被安排处理下一个任务，因此能够避免多次穿件线程，从而节省线程创建和销毁的开销，能带来更好的性能和系统稳定性。<br>
> `线程池设置`：服务器CPU核数有限，能够同时并发的线程数有限，并不是开得越多越好。线程切换是有开销的，如果线程切换过于频繁，反而会使性能降低。假设N核服务器，通过执行业务的单线程分析出本地计算时间x，等待时间为y，则工作线程数设置为 N*(x+y)/x，能让CPU的利用率最大化。

从python3.2开始，标准库提供了`concurrent.futures`模块，它提供了`ThreadPoolExecutor`和`ProcessPoolExecutor`两个类，实现了对`threading`和`multiprocessing`的进一步抽象。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 参数times用来模拟网络请求的时间
def get_html(times):
    time.sleep(times)
    print("get page {}s finished".format(times))
    return times

executor = ThreadPoolExecutor(max_workers=2)
urls = [3, 2, 4] # 并不是真的url

# 方法：as_completed
all_task = [executor.submit(get_html, url) for url in urls]
for future in as_completed(all_task):
    data = future.result()
    print("in main: get page {}s success".format(data))
# 执行结果
# get page 2s finished
# in main: get page 2s success
# get page 3s finished
# in main: get page 3s success
# get page 4s finished
# in main: get page 4s success

# 方法：map
for data in executor.map(get_html, urls):
    print("in main: get page {}s success".format(data))
# 执行结果
# get page 2s finished
# get page 3s finished
# in main: get page 3s success
# in main: get page 2s success
# get page 4s finished
# in main: get page 4s success

```
`ThreadPoolExecutor`构造实例的时候，传入`max_workers`参数：来设置线程池中最多能同时运行的线程数目。

|常用方法||
|:---|:----|
|submit()|用来提交线程池需要执行的任务到线程池中，并返回该任务的句柄，注意：submit不是阻塞的，而是立即返回。<br>`任务句柄`能够使用`done()`方法来判断该任务是否结束。|
|cancel()|可以取消提交的任务。如果任务已经在线程池中运行了，就取消不了了。|
|result()|获取任务的返回值，这个方法内部是阻塞的。|
|as_completed()|判断线程池中那些任务结束了。as_completed方法是一个生成器，在没有任务完成的时候，会阻塞；在有任务完成时，会yield该任务，然后继续阻塞|
|map()||
|wait()||



## 三、多进程

### 1、`multiprocessing模块`

`multiprocessing模块`是python中的多进程管理包，与thread.Thread类似，可以利用`multiprocessing.Process`对象来创建一个进程。该Process对象与Thread对象的用法相同，也有start()，run()，join()方法。

> 1. 在unix平台上，在某个进程结束之后，该进程需要被其父进程调用wait，否则进程成为僵尸进程(zombie)，所以，有必要对每个Process对象调用join方法(等同于wait)。
> 2. `multiprocessing模块`提供了threading包没有的IPC(比如Pipe和Queue)，效率更高，应该优先考虑Pipe和Queue，避免使用Lock/Event/Semaphore/Condition等同步方式。
> 3. 多进程应该避免共享资源。多线程本来就共享资源，可以方便的使用全局变量。各进程有自己的独立空间，共享资源会降低程序的效率。对于多进程，可以通过Manager方法来共享资源。

#### 多进程：

```python
import multiprocessing

q_input = multiprocessing.Queue(100)
q_output = multiprocessing.Queue(100)

all_task = []
for i in range(10):
    all_task.append(multiprocessing.Process(target=函数名, args=(形参)))

for p in all_task:
    p.daemon = True
    p.start()

for p in all_task:
    p.join()

```

#### 进程池(Process Pool)
进程池可以创建多个进程，这些进程就像随时待命的士兵，准备执行任务，一个进程池中可以容纳多个待命的进程。如下：Pool创建了一个容许5个进程的进程池，每个进程都执行f函数，利用map方法将f()函数作用到表的每个元素上。
```python
import multiprocessing

def f(x):
    return x**2

pool = multiprocessing.Pool(processes=5)

#-------map-------#
result = pool.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(result)

#------apply_async--------#
all_task = []
for i in range(1, 10):
    # 进程池中维持processes=5个进程
    all_task.append(pool.apply_async(f, (i,)))

pool.close()
pool.join()
# 结果
result = []
for res in all_task:
    result.append(res.get())

```
|方法|说明|
|:---|:---|
|apply_async(func, args=())|从进程池中取出一个进程执行func函数，args为该函数的参数，它将返回一个AsyncResult对象，可以用该对象的get()方法来获取结果。非阻塞|
|close()|进程池不能再创建新的进程|
|join()|wait进程池中的全部进程，必须对Pool先调用close()方法才能join|

#### 共享内存
可以使用`Value`或`Array`将数据存储在共享内存映射中
```python
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    # d: 表示双精度浮点数据
    # i: 表示有符号整数
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
# 结果
# 3.1415927
# [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
```

#### 服务进程
Manager() 返回的管理器对象控制一个服务进程：用来保存Python对象并允许其他进程使用代理操作他们。利用Manager()可以通过共享进程的方法共享数据。<br>管理器支持的数据类型有：list、dict、Namespace、Lock、RLock、Semaphore、BoundedSemaphore、Condition、Event、Barrier、Queue、Value 和 Array

```python
from multiprocessing import Process,Manager
def func1(shareList,shareValue,shareDict,lock):
    with lock:
        shareValue.value+=1
        shareDict[1]='1'
        shareDict[2]='2'
        for i in xrange(len(shareList)):
            shareList[i]+=1

if __name__ == '__main__':
    manager=Manager()
    list1=manager.list([1,2,3,4,5])
    dict1=manager.dict()
    array1=manager.Array('i',range(10))
    value1=manager.Value('i',1)
    lock=manager.Lock()
    proc=[Process(target=func1,args=(list1,value1,dict1,lock)) for i in xrange(20)]
    for p in proc:
        p.start()
    for p in proc:
        p.join()
    print list1
    print dict1
    print array1
    print value1
# 结果
# [21, 22, 23, 24, 25]
# {1: '1', 2: '2'}
# array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Value('i', 21)

```


### 2、`ProcessPoolExecutor模块`

ProcessPoolExecutor在使用上和ThreadPoolExecutor大致一样，在futures中的方法也是相同的，但对于map()方法，ProcessPoolExecutor会对一个参数`chunksize`，将迭代对象切成块，将其作为分开的任务提交给pool，对于很大的iterables，设置较大的chunksize可以提高性能。

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 参数times用来模拟网络请求的时间
def get_html(times):
    time.sleep(times)
    print("get page {}s finished".format(times))
    return times

executor = ProcessPoolExecutor(max_workers=2)
urls = [3, 2, 4] # 并不是真的url

# 方法：as_completed
all_task = [executor.submit(get_html, url) for url in urls]
for future in as_completed(all_task):
    data = future.result()
    print("in main: get page {}s success".format(data))
# 执行结果
# get page 2s finished
# in main: get page 2s success
# get page 3s finished
# in main: get page 3s success
# get page 4s finished
# in main: get page 4s success

# 方法：map
for data in executor.map(get_html, urls):
    print("in main: get page {}s success".format(data))

```


