轻量化ai编译器
============

本项目目的在于以最简的方式实现一个全流程的ai编译器/运行时,
这套技术有很多种方法可以实现, 在不严重影响性能的前提下,
会尽量挑选代码和逻辑最简单的方案.


技术选型
-------

- 使用python来实现, 因为python代码读写更方便. 尽量使用最新的python.
- 依赖pytorch, 使用pytorch来接入更多常见模型.

开发计划
-------

### 项目结构

- [x] CI script.
    - ![CI status](https://github.com/0x00-pl/plai/actions/workflows/ci.yml/badge.svg?branch=master)

### 接口相关

- [x] 引入pytorch
- [x] 接入torch compile
- [x] 从函数地址解析函数
- [x] 在训练时也调用自定义的compiler
- [x] 解析出计算图

### 算子定义相关

- [x] 定义新的graph/node格式
- [x] node定义中添加namespace信息
- [x] 添加torch和aten的namespace, 覆盖简单模型.
- [ ] node中添加location信息
- [ ] ~~添加多输出的支持~~ (输出tuple的情况不能总是转换成多个输出. 考虑输出是dict的情况)
- [ ] 添加sub-graph的支持

### 运行时相关

- [ ] 添加numpy的namespace, 覆盖简单模型, 用于实现运行时.

### 其他

- [ ] 编译简单的四则运算
- [ ] 简单的四则运算运行时

commands
--------

testing:

```shell
pytest tests
```

generate requirements.txt:

```shell
poetry export --format requirements.txt --output requirements.txt --without-hashes
```

