#   电力领域中文分词模型
[![Build Status](https://travis-ci.org/rises-tech/ecws.svg?branch=master)](https://travis-ci.org/rises-tech/ecws)
[![codecov](https://codecov.io/gh/rises-tech/ecws/branch/master/graph/badge.svg)](https://codecov.io/gh/rises-tech/ecws)
[![Pypi](https://img.shields.io/pypi/v/ecws.svg)](https://pypi.org/project/ecws)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
[![Documentation Status](https://readthedocs.org/projects/ecws/badge/?version=latest)](http://rises-tech.readthedocs.io/?badge=latest)

ecws　是面向电力领域的基础中文分词模型组件，目标是打造电力领域的自然语言处理基础能力

## 安装指南
ecws 依赖以下包:

+ torch==1.5.1
+ allennlp==1.0.0


## 版本号
```version
R3.1
```

## 模型命名
    NLP-ECWS-R3.1

## 使用方式

```
git clone http://114.215.139.32:3000/sbc/NLP-ECWS-R3.1.git
cd NLP-ECWS-R3.1
pip install -e .
```
下载`bert-model`和`model.tar.gz`文件，后续提供

python引用方式：

```python

from ecws.predict import Predictor

bert_model_path = bert_model_path
model_tar_gz_path = model_tar_gz_path

predict = Predictor(model_tar_gz_path, bert_model_path)

d = predict.seg(sent)
```

其中返回的结果是一个字典，字段'sent'中包含分词结果。


## 开发者


