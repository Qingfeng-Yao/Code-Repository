## 参考资料
- 2017 | ICLR | Semi-Supervised Classification with Graph Convolutional Networks | Thomas N. Kipf and Max Welling
- [tkipf/gcn](https://github.com/tkipf/gcn)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 默认数据集`cora`
    - `python3 train.py`: 运行前需要将含有`gcn`前缀的import语句进行修改，即去掉前缀
- 数据集格式: 以`cora`举例
    - ind.cora.x: csr矩阵
    - ind.cora.y: numpy数组
    - ind.cora.tx: csr矩阵
    - ind.cora.ty: numpy数组
    - ind.cora.allx: csr矩阵
    - ind.cora.ally: numpy数组
    - ind.cora.graph: defaultdict，其中key为结点索引，value为其对应的邻居结点列表；之后转换成稀疏邻接矩阵
    - ind.cora.test.index
- 指标: Accuracy

## 数据集统计
| Dataset | #Nodes | #Edges | #Classes | #Features | #Real train | #All train | #Val | #Test | Label rate(#Real train/#Nodes) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Cora | 2708 | 5429 | 7 | 1433 | 140 | 1708 | 500 | 1000 | 0.052 | 

## 运行结果
| Model | Cora | 
| :----: | :----: |
| GCN | 0.816 |