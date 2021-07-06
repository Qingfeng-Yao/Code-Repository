## 参考资料
- 2020 | KDD | MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation | Manqing Dong et al.
- [dongmanqing/Code-for-MAMO](https://github.com/dongmanqing/Code-for-MAMO)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 
    - 数据准备: 
        - 在`data_raw`下的两个目录下分别放置原始数据
        - 创建目录`data_processed`
        - 运行`python3 prepareDataset.py`: 共处理两个数据集(MovieLens和Bookcrossing)
            - MovieLens: 用户特征(ID，性别，年龄，职业/21种)，item特征(ID，rate，年份，类型，导演)，打分(用户ID，itemID，分数1-5，时间戳)
            - 限制用户的交互数为20，且依据评分时间划分用户为warm和cold
            - 依据评分数划分item为warm和cold
            - 特征为索引则随机初始化嵌入(nn.Embedding)，特征有多个取值则使用one-hot编码，然后变换(nn.Linear)成和其他嵌入一样的维度
    - `python3 mamoRec.py`
        - 随机初始化特征，包括用户三个特征和item四个特征
        - 嵌入层(MLP)
        - 推荐模型(将用户嵌入和item嵌入连接作为输入，使用一个用户偏好内存矩阵，即维度不变的线性变换，再使用MLP)
        - 随机初始化三个内存矩阵，前两个与用户特征相关，后一个与任务相关
        - 训练: 局部参数初始化(包括初始化用户偏好内存矩阵，需计算初始化参数的偏差项)和更新(包括三个内存矩阵的更新)；全局参数更新

## 数据集MovieLens 1M统计信息
| # origin users | # origin items | # origin ratings | # processed users | # warm users | # cold users | # processed items | # warm items | # cold items | # processed ratings | # user dim | # item dim | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 6040 | 3881 | 1000209 | 6040 | 5400 | 640 | 3328 | 1683 | 1645 | 120800 | 3(2+7+21) | 2+25+2186(6+1+25+2186) | 