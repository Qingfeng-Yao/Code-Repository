## 参考资料
- [Open Graph Benchmark](https://ogb.stanford.edu/)
- [ogb-example](https://github.com/snap-stanford/ogb/tree/master/examples)
- 2020 | Open Graph Benchmark: Datasets for Machine Learning on Graphs | Weihua Hu et al.

## Open Graph Benchmark
- 图机器学习的基准数据集(benchmark datasets)、数据下载(data loaders)以及评估(evaluators)
- data loaders包括数据集的下载、预处理
- 环境配置: 在Pytorch Geometric环境的基础上`pip install ogb`; 确保版本是1.3.1
- 运行`get_start_ogb.ipynb`: 数据下载和标准化模型评估
- ogb例子
    - 结点属性预测任务
        - arxiv: `python3 gnn.py`
    - 链接属性预测任务
    - 图属性预测任务