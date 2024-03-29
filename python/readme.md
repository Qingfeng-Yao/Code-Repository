### basic
- python基础
- 命令行
- 深度学习
- 参考资料
    - [taizilongxu/interview_python](https://github.com/taizilongxu/interview_python)
    - [kenwoodjw/python_interview_question](https://github.com/kenwoodjw/python_interview_question)
    - [chiphuyen/python-is-cool](https://github.com/chiphuyen/python-is-cool)
    - [rougier/numpy-100](https://github.com/rougier/numpy-100)
    - [动手学深度学习](http://zh.d2l.ai/)

### leetcode
- 具体内容
    - 数组: 基础, 矩阵, 其他(包括三角形, 数学, 约束限制和空间优化)
    - 子序列: 最值, 遍历
    - 字符串: 遍历, 正则
    - 二进制: 异或, 二进制运算
    - 树: 递归, 迭代
    - 链表: 递归, 迭代, 创建
    - 排序
    - 回溯: 排列组合
    - 设计: 哈希, 缓存, 数据结构
    - 其他: 拒绝采样
- 参考资料
    - [CodeTop](https://codetop.cc/#/home)
    - [geekxh/hello-algorithm](https://github.com/geekxh/hello-algorithm)
    - [leetcode多种解法](https://leetcode.wang/)
    - [算法工程师面试题整理-数据结构与算法部分](https://github.com/PPshrimpGo/AIinterview)

### repo
- python中一些常用的库
- 参考资料
    - [城东-特征工程](https://www.zhihu.com/question/29316149/answer/110159647)
    - [各种文本分类算法集锦，从入门到精通](https://www.heywhale.com/mw/project/5be7e948954d6e0010632ef2/content)

### QA
- ERROR: Could not find a version that satisfies the requirement yaml
    - 应安装`pyyaml`
- ImportError: No module named 'cPickle'
    - python2有cPickle，但是在python3下，是没有cPickle的
    - 将cPickle改为pickle即可
- ImportError: cannot import name 'create_T_one_hot'
    - 交叉import: a,b两个python文件，在a中import b中的类，又在b中import a中的类，就会报这种异常
- SyntaxError: Non-UTF-8 code starting with '\xbb' in file quick_start.py on line 13, but no encoding declared
    - 当python文件中出现中文
    - 解决办法为: 在文件首行添加: `# -*- coding:utf-8 -*-/# _*_ coding: utf-8 _*_`
- TypeError: can't multiply sequence by non-int of type 'float'
    - `'*'*5.0`
    - '*'*5则没有问题
