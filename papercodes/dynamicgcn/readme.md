## 参考资料
- 2019 | KDD | Learning Dynamic Context Graphs for Predicting Social Events | Songgaojun Deng et al.
- [amy-deng/DynamicGCN](https://github.com/amy-deng/DynamicGCN)

## 源码运行
- 环境配置: `python3.7`; `requirements.txt`
    - pytorch_sparse的安装: `python3 -m pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html`
- 命令运行: `python3 train.py`(默认数据集THAD6h)
    - 可选命令: `python3 train.py --dataset INDD6h --embedding india`
    - 目前可获得的数据集为THAD6h、INDD6h、EGYD6h和RUSD6h
        - `ind.*.idx / ind.*.tidx`: 用于训练/测试的词索引文件；`ind.*.x / ind.*.tx`: 用于训练/测试的时序图输入文件；`ind.*.y / ind.*.ty`: 用于训练/测试的ground truth
        - 四个数据集对应的嵌入文件thailand.emb_100、india.emb_100、egypt.emb_100、russian.emb_100；100维预训练word2vec嵌入
    - 所有参数使用Glorot initialization初始化；为了使模型收敛得更快，使用批标准化
    - train/0.825/val/0.175；历史跨度为7；针对不同数据集设置不同的类权重
    - 源码问题:
        - utils.py中出现`ValueError: Object arrays cannot be loaded when allow_pickle=False`: 将np.load的allow_pickle参数改为True
        - utils.py中出现`FileNotFoundError: [Errno 2] No such file or directory: 'data/thailand.emb_100'`: 将嵌入文件目录改为'data/{}'.format(dataset_str)
        - layer.py中出现`RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasCreate(handle)`，做矩阵乘法时报错，将pytorch的版本从1.8.0换成1.7.1
        - layer.py中出现`TypeError: spmm() missing 1 required positional argument: 'matrix'`: spmm函数添加参数adj.size(1)
- 事件预测任务
- 数据集
    - 论文中使用来自Integrated Conflict Early Warning System(ICEWS)的事件数据，它包含政治事件，目的在于评估国内和国际的危机事件。这些事件被编码成20个主要类和它们的子类。每个事件具有地理(城市，地区，国家)、时间(年、月、日)、类别以及相关联的文本。
    - 论文主要关注事件的一个主要类(抗议)，并从四个国家中选择数据集，包括印度、埃及、泰国和俄罗斯
    - 数据预处理: 对于每个城市，以事件之前的k天内的文档作为原始输入，以目标事件是否发生作为ground truth
        - cleaning and tokenizing words-->remove stop words and keep only stemmed words
        - 词汇: 移除少于5次的低频词和文档频率高于80%的词
        - 抛弃文章数特别少的样本
        - 所有数据集的正负样本数3:5
        - 平均每个图的结点数为600左右
        - 每个例子的结点数不同
- 评估指标
    - 预测性能: Precision (Prec.), Recall (Rec.), and F1-Score (F1)

## 数据集统计
- 正样本与负样本的数目的与论文的数据稍微有出入
|  | Thailand | Egypt | India | Russia | 
| :----: | :----: | :----: | :----: | :----: |
| samples | 1883 | 3788 | 12249 | 3552 | 
| train | 1600 | 3219 | 10411 | 3019 | 
| test | 283 | 569 | 1838 | 533 | 
| pos | 707 | 1489 | 4548 | 1178 | 
| neg | 1176 | 2299 | 7701 | 2374 | 
| vocabulary | 27281 | 19680 | 75994 | 49776 | 


## 运行结果
- 学习率与论文设置的不同；早停止设置为10
|  | Thailand | Egypt | India | Russia | 
| :----: | :----: | :----: | :----: | :----: |
| F1 | 0.8111 | 0.8610 | 0.6766 | 0.7859 | 
| Rec. | 0.7766 | 0.8043 | 0.6477 | 0.8619 | 
| Prec. | 0.8488 | 0.9265 | 0.7082 | 0.7222 | 