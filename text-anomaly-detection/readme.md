## 任务
- one-class分类

## 测试指标
- AUC

## 数据集
- Reuters-21578: 共7类 --> nltk
    - 预处理后训练集：7769；测试集：3019
    - 只考虑单标签数据
- 20 Newsgroups: 共6类 --> sklearn
    - 预处理后训练集：10996；测试集：7299
- IMDB: 共2类 --> torchnlp
    - 预处理后训练集：25000；测试集：25000
- 正常数据标记为0，异常数据标记为1
- 文本预处理：小写-->移除标点/数字/空格/停用词/长度小于3的词
- torchnlp.datasets.dataset.Dataset-->torch.utils.data.DataLoader

## 预训练模型
- FastText_en:默认使用
- GloVe_6B
- bert-base-uncased
    - bert使用需要限制文本输入序列长度不超过512，且需要减小batch_size；bert使用效果不佳
- 嵌入不更新

## CVDD模型异常检测
- 文本表示：利用自注意力机制获得文本表示
- 利用与上下文向量的距离进行异常检测

## 流模型异常检测
- 文本表示
    - 嵌入均值、嵌入最大值、嵌入的tfidf加权和
    - 基于RNN的条件流: 使用GRU、双向
- 流模型
    - MAF



## 参考资料
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows)
- [zalandoresearch/pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)
- [yao8839836/text_gcn](https://github.com/yao8839836/text_gcn)