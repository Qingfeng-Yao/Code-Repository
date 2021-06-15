## 参考资料
- [graph4ai/graph4nlp](https://github.com/graph4ai/graph4nlp)
- [graph4nlp的参考文档](http://saizhuo.wang/g4nlp/index.html)
- [graph4nlp的参考文献](https://github.com/graph4ai/graph4nlp_literature)

## 内容
- 自然语言处理与图深度学习的结合，目的在于将GNN应用于NLP任务
- 建立在`DGL`上，包含四层: 数据层、模块层、模型层、应用层
- 测试运行
	- 环境配置: `python3.6`; `requirements.txt`
	- 运行`quick2Graph2seq.ipynb`: 其中job数据集无法下载，需要自行整理到raw目录下；另外还需要下载任务参数到examples目录下
		- 运行前还需要先运行[stanfordcorenlp](https://stanfordnlp.github.io/CoreNLP/download.html)，否则会出现数据无法下载的情况，即`jobs_dataset.train`为空；目前仍有这个问题
		- 熟悉jobs数据集和graph2seq模型