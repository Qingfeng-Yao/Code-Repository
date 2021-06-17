## 参考资料
- 2019 | ICML | Latent Normalizing Flows for Discrete Sequences | Zachary M. Ziegler and Alexander M. Rush
- [harvardnlp/TextFlow](https://github.com/harvardnlp/TextFlow)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 字符级别语言建模实验
	- 运行基线模型LSTM: `python3 main.py --dataset ptb --run_name charptb_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20`
	- 运行Text-Flow(AF-AF): `python3 main.py --dataset ptb --run_name charptb_discreteflow_af-af`
- 指标: 负对数似然(bpc)

## 运行结果
|  | PTB |
| :----: | :----: |
| LSTM | 
| TextFlow(AF-AF) | 