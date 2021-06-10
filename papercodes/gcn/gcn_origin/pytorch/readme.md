## 参考资料
- 2017 | ICLR | Semi-Supervised Classification with Graph Convolutional Networks | Thomas N. Kipf and Max Welling
- [tkipf/pygcn](https://github.com/tkipf/pygcn)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 默认数据集`cora`
    - `python3 train.py`: 运行前需要将含有`pygcn`前缀的import语句进行修改，即去掉前缀；修改函数`load_data`的路径参数为`data/cora/`
        - 遇到问题: 运行到model.cuda()卡死，原因是cuda与pytorch不匹配，我原先使用的torch是0.4，我的cuda是10.1，可使用torch==1.6.0
- 数据集格式: 以`cora`举例
    - cora.cites: 存储边的信息
    - cora.content: 存储结点ID、结点特征及结点标签信息
- 指标: Accuracy

## 数据集统计
| Dataset | #Nodes | #Edges | #Classes | #Features | #Real train | #Val | #Test | Label rate(#Real train/#Nodes) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Cora | 2708 | 5429 | 7 | 1433 | 140 | 300 | 1000 | 0.052 | 

## 运行结果
| Model | Cora | 
| :----: | :----: |
| GCN | 0.827 |