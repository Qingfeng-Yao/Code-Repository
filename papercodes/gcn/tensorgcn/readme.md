## 参考资料
2020 | AAAI | Tensor Graph Convolutional Networks for Text Classification | Xien Liu et al.
- [THUMLP/TensorGCN](https://github.com/THUMLP/TensorGCN)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 默认数据集为mr
    - `python3 build_graph_tgcn.py`: 基于序列的图与textgcn一样处理；此外还构建了基于语义的图和基于句法的图，如果不存在句法或语义关系，则用pmi值填充；上述都是词与词之间的关系，词与文档之间的关系则均用tfidf值
    - `python3 train.py`: 运行时需提前创建`/data_tgcn/mr/build_train/mr_best_models/`，然后将训练文件中的istrain参数改为True；训练结束后，再将istrain参数改为False进行测试
- 数据集格式: 以mr举例
    - `mr.clean.txt`为每个文档的文本内容
    - `mr.txt`为文档的名字、对应的划分以及标签
    - `mr_pair_stan.pkl`包含数据集中所有具有句法关系的词对
    - `mmr_semantic_0.05.pkl`包含数据集中所有具有语义关系的词对
- 指标: Accuracy

## 数据集统计
| Dataset | #Docs | #Training | #Real Training(90%) | #All Training(Words) | #Test | #Words | #Nodes | #Feature Dim | #Classes |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| MR | 10662 | 7108 | 6398 | 25872 | 3554 | 18764 | 29426 | 300 | 2 |

## 运行结果
| Model | MR | 
| :----: | :----: |
| TensorGCN | 0.76928 |