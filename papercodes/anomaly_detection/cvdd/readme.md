## 参考资料
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)

# 源码运行
- 环境配置: `python3.7`; `requirements.txt`
	- 还需要下载`spaCy en`库: `python3 -m spacy download en`;该命令会出现错误`requests.exceptions.ConnectionError`，目前只能自己下载好en目录并放在`python3.7/site-packages/spacy/data`目录下
- 命令运行: 先创建`data`目录，然后进入`src`目录
	- Reuters-21578
		- 先创建目录`../log/test_reuters`
		- 正常类为`ship`(索引为6)，使用GloVe_6B为词嵌入，3个注意力头，注意力大小为150: `python3 main.py reuters cvdd_Net ../log/test_reuters ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 6`
			- 索引[0, 1, 2, 3, 4, 5, 6]分别对应['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']
		- 直接运行上述命令会出现问题，要求先下载好数据集reuters: `import nltk; nltk.download('reuters')`；此时会出现问题`Connection refused`，目前是先下载好数据集然后存放地址为`/data/corpora/reuters.zip`。`stopwords`和`punkt`类似，目前是先下载好数据集然后存放地址为`/data/corpora/stopwords.zip`和`/data/corpora/stopwords`以及`/data/tokenizers`
	- 20 Newsgroups
		- 先创建目录`../log/test_newsgroups20`
		- 正常类为`comp`(索引为0)，使用FastText_en为词嵌入，3个注意力头，注意力大小为150: `python3 main.py newsgroups20 cvdd_Net ../log/test_newsgroups20 ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
			- 索引[0, 1, 2, 3, 4, 5]分别对应['comp', 'rec', 'sci', 'misc', 'pol', 'rel']
		- from torchtext.vocab import FastText
		- 下载`wiki.en.vec`会遇问题: `urllib.error.HTTPError: HTTP Error 403: Forbidden`
- 评估指标: AUC

## 运行结果
Reuters-21578
| earn | acq | crude | trade | money-fx | interest | ship | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 93.96 | 90.13 | 89.73 | 97.93 | 81.90 | 92.53 | 96.63 | 

20 Newsgroups
| comp | rec | sci | misc | pol | rel | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| 74.18 | 59.87 | 58.05 | 75.23 | 71.03 | 77.75 | 
