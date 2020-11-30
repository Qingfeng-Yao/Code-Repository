## 任务
- one-class分类

## 数据集
- Reuters-21578: 共7类 --> nltk
    - 预处理后训练集：7769；测试集：3019
    - 只考虑单标签数据
- 20 Newsgroups: 共6类 --> sklearn
    - 预处理后训练集：10996；测试集：7299
- 文本预处理：小写-->移除标点/数字/空格/停用词/长度小于3的词
- torchnlp.datasets.dataset.Dataset-->torch.utils.data.DataLoader

## 模型
- 预训练模型
    - FastText_en
    - GloVe_6B
    - bert-base-uncased
    - 更新及标准化嵌入
- 文本表示
    - 嵌入均值
    - 嵌入最大值
    - 嵌入的tfidf加权和
- 流模型
    - MAF
    - MAF-split
    - MAF-glow
    - MAF-split-glow

## 测试指标
- AUC
- 测试集计算似然值时可能会出现极小极大值，使用np.nan_to_num解决

## 参考资料
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows)