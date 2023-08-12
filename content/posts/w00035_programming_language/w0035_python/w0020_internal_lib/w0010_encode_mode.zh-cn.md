---
title: "字符编码"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: 字符编码
    identifier: python-encode
    parent: python-internal
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","字符编码"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、字符编码

> <mark>ASCII</mark>：计算机是美国人发明的，所以最早只考虑了简单的26个字母和一些控制字符，所以只用7-bit组合出128个组合，编号0~127，存储的时候凑成了一个byte。这个组合没有考虑其他国家，比如汉字就不只128个，于是中国为汉字编码发明了GB2312编码，其他国家也有自己的各种编码，互不兼容。<br> 为了统一，提出了<mark>unicode编码</mark>，包含了各个国家的文字，对每个字符都用2个byte来表示，英文的话就在前面加0。<br>unicode对于英文就会有些浪费，为了解决这个问题，为了节约硬盘空间/ 网络带宽，又发明了<mark>utf-8编码</mark>，1个字符可能会被编码成1~6个字节，英文还是1个字节，汉字变成了3个字节，只有在生僻字才会在4个字节。

|字符|ASCII|unicode|utf-8|
|:--|:--|:--|:--|
|A|01000001|00000000 01000001|01000001|
|中||01001110 00101101|11100100 10111000 10101101|
|字符应用层的形式||字符在内存的形式|字符在硬盘/网络中的形式|


## 二、解析/转换

图片在网络中获取下来是二进制的格式(bytes)；或者通过 open('***.jpg', 'rb') 读取的图片也是二进制的格式<br>

1. `bytes格式 <-> str`<br>
    - bytes: 是(二进制)数字序列，是utf-8的编码形式。该格式的变量是不可修改的。
        - str --> bytes : 使用str.encode()方法
        - bytes --> str : 使用bytes.decode()方法 
    - bytearray(): 该格式的变量是可以修改的 
    ```python
    a = '人生苦短'
    # 此时b的格式是bytes，是不能修改的，即不能操作：b[:6] = '生命'.encode() 
    b = a.encode() # \xe4\xba\xba\xe7\x94\x9f\xe8\x8b\xa6\xe7\x9f\xad
    c = bytearray(b) # 转变为bytearray格式，就可以修改了
    c[:6] = bytearray('生命'.encode())
    ```

2. `bytes格式 <-> numpy` <br>

    - bytes --> numpy : 
    ```python
    img_np = np.asarray(bytearray(content), dtype='uint8')
    # 或者
    img_np = np.frombuffer(content, dtype='uint8')
    ```
    - numpy --> bytes : `img_content = img_np.tobytes()`

3. `bytes格式 <-> PIL`

    - bytes --> PIL : 
    ```python
    content = b'\x...' # 二进制序列 utf-8编码格式
    img_pil = PIL.Image.open(BytesIO(content))
    ```
    - PIL --> bytes :
    ```python
    from PIL import Image
    from io import BytesIO
    # BytesIO: 在内存中读写bytes. 
    # 例如：f = BytesIO()  f.write('中文'.encode())  f.getvalue()
    img_pil = Image.open('***.png')
    f = BytesIO()
    img_pil.save(f, format='PNG')  # PNG参数：四通道；JPEG参数：三通道
    img_bytes = f.getvalue()  # 转二进制  
    ```

4. `bytes格式 <-> opencv`
    - bytes -> cv2
    ```python
    img_np = np.asarray(bytearray(content), dtype='uint8')
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    ```
    - cv2 -> bytes
    ```python
    success, encode_img = cv2.imencode('.jpg', img_cv)
    img_bytes = encode_img.tostring()
    ```


5. `PIL <-> np`
    - PIL -> np
    ```python
    img_np = np.array(img_pil)
    ```
    - np -> PIL
    ```python
    img_pil = Image.fromarray(img_np)
    ```

6. `PIL <-> cv2` PIL的图片是RGB模式，cv2的图片是BGR格式
    - PIL -> cv2
    ```python
    img_cv = cv2.cvtColor(numpy.asarray(img_pil), cv2.COLOR_RGB2BGR)
    ```
    - cv2 -> PIL
    ```python
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    ```

