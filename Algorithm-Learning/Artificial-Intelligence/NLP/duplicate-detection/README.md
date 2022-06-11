## 文本去重
### 识别相似文档
- 最终目标
	- 识别Web mirrors
	- spam detection
	- semantic similarity
- 所考虑的语料库
	- 网页文档
	- 文件系统中的文件
	- E-mails
	- 特定领域的语料库
- 每篇文档所对应的特征集: 如文档向量
- 压缩特征集的签名方案

### semantic similarity
- knowledge-based methods：借助潜在的知识源，如词汇数据库；考虑了实际的词义
	- 词汇数据库
		- WordNet: 超过10万个英语概念；可视化为图，节点表示词义(概念)，边定义词之间的关系；WordNet的结构主要基于同义词
		- Wiktionary: 包含大约620万个词(来自4000种语言)
		- Wikipedia
		- BabelNet
- corpus-based methods：借助一个大的潜在语料库(词嵌入，相似的词频繁地在一起出现，但实际的词义没有考虑)
- deep neural network-based methods
- hybrid methods

### 算法
- simhash |[c++](https://github.com/yanyiwu/simhash)| [python](https://github.com/leonsim/simhash)
	- Charikar’s fingerprinting technique
	- 适用于长文本及文本差异很小的情况(在64位的情况下将海明距离设为3；即几乎重复的文档只在少量比特位置上不同)
- tfidf-lsa

### 指标
* precision
* recall
* f1-score
* 计算方法：对于每一个文档执行一遍去重查询(不包括本身)，与提供的数据标准结果进行比较计算，最后求平均值(若去重结果为空，则p和r都置为1)(f1-score可基于p和r的平均值来求)

### learning to hash (L2H)
- 能够学习为给定数据集定制的保持相似性的哈希函数(海明距离计算相似性)

### 参考文献
- 2002 | Similarity estimation techniques from rounding algorithms | M. Charikar
- 2007 WWW | Detecting Near-Duplicates for Web Crawling | Gurmeet Singh Manku, Arvind Jain and Anish Das Sarma
- 2018 SIGMOD | A General and Efficient Querying Method for Learning to Hash | Jinfeng Li et al.
