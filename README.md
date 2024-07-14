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

- [x] 引入pytorch
- [x] 接入torch compile
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

