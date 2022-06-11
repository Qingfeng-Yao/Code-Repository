## 环境依赖与实验设置
- conda: cvdd/cnf(df模型)
- 实验遵循2019Lukas Ruff等人的实验设置
    - 一元分类实验: 某一个类是正常类，剩余类被认为是异常；只用正常类的数据来训练模型，使用所有类的数据进行测试，正常类的标签为0，异常类的标签为1
    - 指标: AUC

## 数据集
- Reuters-21578
    - 该数据集是多标签，考虑只有一个标签的样本
    - 最小词频为10
- 20 Newsgroups
    - 最小词频为20
- IMDB Movie Reviews
    - 最小词频为3
- 数据预处理: 小写；去除标点、数字和多余的空格；去除停用词/nltk；考虑长度至少为3个字符的词

## 预训练模型
- GloVe: 6B tokens vector embeddings of p = 50 dimensions

## 模型
- CVDD: 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.
    - 注意力大小为150，注意力头数目为3
    - 使用Adam，批大小为64
    - 训练时需要指明dataset和normal_class，必要时还需要指明min_count
- Discrete Flows(自回归): 2019 | Discrete Flows: Invertible Generative Models of Discrete Data | Dustin Tran et al.
    - `python3 main_df.py --min_count 20 --dataset REUTERS_DATA --normal_class 0`
- Categorical Normalizing Flows: 2020 | Categorical normalizing flows via continuous transformations | Phillip Lippe and Efstratios Gavves
    - `python3 main_cnf.py --max_seq_len 450 --batch_size 64 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 64 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --min_count 20 --dataset REUTERS_DATA --normal_class 0`
- EmbeddingNF: 基于文本嵌入的标准化流
    - `python3 main_embflow.py --max_seq_len 450 --batch_size 8 --encoding_dim 50 --coupling_hidden_layers 1 --coupling_num_mixtures 64 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --min_count 20 --dataset REUTERS_DATA --normal_class 0`
- 模型训练: 总共训练100epochs；若验证损失连续30epochs不下降，则停止训练；同时保存验证损失最低的模型

## 参考资料
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [google/edward2](https://github.com/google/edward2)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)