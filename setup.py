from setuptools import setup, find_packages
LONGDOC = """
ecws
=====
电力领域中文分词：适应于电力行业的中文分词组件
Chinese word segmentation in the power industry: Chinese word segmentation components adapted to the power industry
完整文档见 ``README.md``
GitHub: https://github.com/campper
安装说明
========
代码对 Python 2/3 均兼容
-  全自动安装： ``easy_install ecws`` 或者 ``pip install ecws`` / ``pip3 install ecws``
-  半自动安装：先下载 https://pypi.python.org/pypi/ecws/ ，解压后运行
   python setup.py install
-  手动安装：将 ecws 目录放置于当前目录或者 site-packages 目录
-  通过 ``import ecws`` 来引用
"""

setup(
    name="ecws",
    version="3.0",
    keywords=("ecws", "3.0"),

    url="http://rises.tech",
    author="qzhang",
    author_email="campxp@hotmail.com",
    description='GLOBAL ENERGY INTERCONNECTION RESEARCH INSTITUTE',
    packages=find_packages(),
    include_package_data=True,
    platforms="any",

    install_requires=[
        "torch==1.5.1",
        "allennlp==1.0.0",
    ]
)
