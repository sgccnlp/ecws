#   电力领域中文分词模型
ecws　是面向电力领域的基础中文分词模型组件，目标是打造电力领域的自然语言处理基础能力

## 安装指南
ecws 依赖以下包:

+ torch==1.5.1
+ allennlp==1.0.0


## 版本号
```
R3.0.5
```

## 模型命名
  NLP-ECWS-R3.0.5

## 安装

* 第一步，安装 ecws

使用 pip 安装
```bash
pip install ecws
```
或从源代码安装

```bash
git clone https://github.com/rises-tech/ecws.git
cd NLP-ECWS
pip install -e .
```
 
* 下载`ecws.model`文件

| Model Name  | Download Link |
| ------------------  |  ---------------  |
| ecws.v3.model | [BaiduPan](https://pan.baidu.com/s/1a6DoMVRdJLdC9gZOJL88aA) 提取码：ecws |



* python引用方式：

```python

from ecws.segment import Segmenter

model_path = 'ecws.model'

predict = Segmenter(model_path)

d = predict.seg(sent)
```
* 接口demo界面
```
http://120.27.25.150:8082/
```

* web api 调用方式

```python
def webservice_ecws(sentence):
  data = {'sent': sentence}
  url = 'http://120.27.25.150:8082/predict'
  r = requests.post(url, json=data)
  data = json.loads(r.text)
  seg = data['spans']
  return seg
```
其中返回的结果是一个字典，字段'sent'中包含分词结果。

* 电力自然语言处理演示平台

[http://demo.rises.tech](http://demo.rises.tech)

## 开发者
@ 张强<<[alxor@live.cn](alxor@live.cn)>>
@ 宋博川<<[abc_hhhh@163.com](abc_hhhh@163.com)>>

