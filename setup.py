# from setuptools import setup,find_packages
import setuptools
from ecws import VERSION

SHORT_DESC = (
    "面向电力领域的中文分词工具"
)

with open("README.md","r") as fh:
    long_description = fh.read()
    

setuptools.setup(
    name="ecws",
    version=VERSION,
    description=SHORT_DESC,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=("ecws", "3.0.2"),
    url="http://github.com/rises-tech/ecws",
    author="alxor",
    author_email="alxor@live.cn",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['README.md', 'LICENSE.txt']},
    zip_safe=False,
    platforms="any",
    install_requires=[
        "torch==1.5.1",
        "allennlp==1.0.0",
    ],
    python_requires='>=3.6'
)
