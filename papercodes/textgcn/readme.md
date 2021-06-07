## 参考资料
- 2019 | AAAI | Graph Convolutional Networks for Text Classification | Liang Yao et al.
- [yao8839836/text_gcn](https://github.com/yao8839836/text_gcn)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 下面的每一条命令中的`20ng`都可替换成`R8`, `R52`, `ohsumed`和`mr`
    - `python3 remove_words.py 20ng`: 读取每个数据集中的每个文档，存储于doc_content_list中；然后分词并统计词频；去除停用词和词频小于5的词(mr不做处理)得到clean_docs
    - `python3 build_graph.py 20ng`: 读取每个数据集中文档对应的划分和标签，可得到doc_name_list,doc_train_list和doc_test_list；从doc_name_list中得到训练和测试样本的索引；统计词汇数、整个语料库的词频、词出现的文档数、标签数；训练集再划分出验证集；构建特征矩阵(文档结点特征为空、词结点特征随机初始化；训练时使用单位矩阵)、标签矩阵(文档结点标签为one-hot、词结点标签为空)；构建词-文档异质图，即邻接矩阵(设置上下文窗口大小为20，在文档中一个词一个词地滑动，遍历所有文档；统计每个词出现的窗口数、每个词对出现的次数(不是窗口数?)；计算pmi；统计每个文档下不同词出现的次数；计算tfidf)
    - `python3 train.py 20ng`: 下载数据，特征矩阵和邻接矩阵预处理；创建模型(两层图卷积网络)
- 数据集: 实验考虑5个数据集，分别是`20ng`, `R8`, `R52`, `ohsumed`, `mr`，数据集内容存储于`data/corpus/`，数据集划分及标签存储于`data/`
    - 数据预处理: 使用nltk的stopwords
- 指标: Accuracy

## 数据集统计
| Dataset | #Docs | #Training | #Real Training(90%) | #All Training(Nodes) | #Test | #Words | #Nodes | #Feature Dim | #Classes | Average Length | Min Length | Max Length | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 20NG | 18846 | 11314 | 10183 | 54071 | 7532 | 42757 | 61603 | 300 | 20 | 221.26 | 14 | 35702 | 

## 运行结果
| Model | 20NG | 
| :----: | :----: |
| Text GCN | 0.8574 |