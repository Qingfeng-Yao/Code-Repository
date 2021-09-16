## 参考资料
### text modeling
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)
- [google/edward2](https://github.com/google/edward2)

### text anomaly detection
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)

## 参考论文
### text modeling
- 2019 | NIPS | Discrete Flows: Invertible Generative Models of Discrete Data | Dustin Tran et al.(Google)
- 2021 | ICLR | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves

### text anomaly detection
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.

## 环境配置
### text modeling
- `python3.6`; `text_modeling/requirements.txt`
    - 可设置conda环境(textgmpy)

### text anomaly detection
- `python3.7`; `text_anomaly_detection/requirements.txt`
    - 下载`spaCy en`库: `python3 -m spacy download en`;可能会因为`timed out`出现无法下载的问题，可多次尝试，实在不行只能利用已经下载好的en目录并放在`python3.7/site-packages/spacy/data`目录下
    - 可设置conda环境(textanopy)

## 任务
### text(language) modeling
#### 数据集
- Penn Treebank
- text8
- 执行每个数据文件中的main函数现将对应数据集下载到`data`文件夹下，`Penn Treebank`限制序列长度为(0,288)，词汇数为51，`text8`限制序列长度为256，词汇数为27，均是字符级别
- `Penn Treebank`训练集中序列总数41801，总字符数4751151；验证集中序列总数3360，总字符数375882；测试集中序列总数3739，总字符数416353 
- `text8`训练集中序列总数5624983，总字符数90000000；验证集中序列总数19529，总字符数5000000；测试集中序列总数19529，总字符数5000000

#### 模型
- LSTM
    - 嵌入层: 从离散到连续
    - 时间嵌入层: 在原始嵌入上扩充嵌入维度
    - LSTM网络
    - 输出层: 输出维度为词汇表大小的线性层
        - LSTM网络经输出层得到的概率为下一个token的预测概率
        - 第一个token的预测概率由初始state向量经输出层得到
- Categorical Normalizing Flow
    - 编码层: 从先验分布中采样数据并进行变换，并以离散数据的嵌入控制变换的参数
    - 激活层: 变换的参数为可学习参数
    - 自回归混合耦合层
- Discrete Flows: 
    - Discrete autoregressive flows
    - Discrete bipartite flows
    - 对输入均进行one-hot编码，然后DAF使用MADE变换，DBF使用恒等变换

#### 相关执行命令
- 统一实验设置: use_multi_gpu/False, restart/False, checkpoint_path/None, load_config/False, no_model_checkpoints/False, only_eval/False, cluster/True, debug/False, clean_up/False
- 指标: 测试集上的负对数似然(test bpd), 生成一个序列所需要的时间(generate, 单位s, PTB对应序列长度为288, text8对应序列长度为256)
- PTB+CNF: `python3 train.py --dataset penntreebank --model_name CNF --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.26
- PTB+RNN: `python3 train.py --dataset penntreebank --model_name RNN --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.273
- PTB+DAF: `python3 train.py --dataset penntreebank --model_name DAF --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --discrete_num_flows 1 --temperature 0.1 --discrete_nh 512 --learning_rate 7.5e-4 --cluster`
    - test bpd: 4.033
- PTB+DBF: `python3 train.py --dataset penntreebank --model_name DBF --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --discrete_num_flows 1 --temperature 0.1 --discrete_nh 512 --learning_rate 7.5e-4 --cluster`
    - test bpd: 4.073
- text8+CNF: `python3 train.py --dataset text8 --model_name CNF --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.448
- text8+RNN: `python3 train.py --dataset text8 --model_name RNN --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.435 
- text8+DAF: `python3 train.py --dataset text8 --model_name DAF --max_iterations 100000 --max_seq_len 256 --batch_size 128 --discrete_num_flows 1 --temperature 0.1 --discrete_nh 512 --learning_rate 7.5e-4 --cluster`
    - test bpd: 4.125
- text8+DBF: `python3 train.py --dataset text8 --model_name DBF --max_iterations 100000 --max_seq_len 256 --batch_size 128 --discrete_num_flows 1 --temperature 0.1 --discrete_nh 512 --learning_rate 7.5e-4 --cluster`
    - test bpd: 4.411

### text anomaly detection
#### 数据集
- 分别统计每个数据集以每个类为正常数据时的训练集(只包含正常数据)、测试集(同时包含正常数据和异常数据)以及词汇表大小，数值在下面每个类别后的括号内
- Reuters-21578
    - 共7类: earn(2840/1083/1097/6714), acq(1596/696/1484/7524), crude(253/121/2059/5623), trade(250/76/2104/5713), money-fx(222/87/2093/5444), interest(191/81/2099/5240), ship(108/36/2144/5272)
    - 测试集的总大小均是2180
- 20 Newsgroups
    - 共6类: comp(2857/1909/5390/25818), rec(2301/1524/5775/23578), sci(2311/1520/5779/24873), misc(577/382/6917/21389), pol(1531/1025/6274/24015), rel(1419/939/6360/22935)
    - 测试集的总大小均是7299
- IMDB
    - 共2类: pos(12500/12500/12500/44424), neg(12500/12500/12500/43442)
    - 测试集的总大小均是25000
- 先创建目录`data`，然后进入`src`，对`Reuters-21578`，执行`import nltk;nltk.download('reuters', download_dir='../data');nltk.download('stopwords', download_dir='../data');nltk.download('punkt', download_dir='../data')`；其余两个数据集以及相应的预训练词向量会自动下载
- 上述文本均经过如下的预处理: 小写；移除标点、前后空格、停用词和长度小于3的词
- 每个样本由字典`{'text': text,'label': label}`构成，标签则根据是否是正类分为0(正常)和1(异常)，其中对`Reuters-21578`，只有单标签样本被考虑
- 训练集均是正常样本，测试集包括所有样本

#### 模型
- CVDD

#### 相关执行命令
- 执行命令前需要先为每个数据集创建日志目录，如`log/test_reuters;log/test_newsgroups20;log/test_imdb`
- CVDD+reuters: `python3 main.py reuters cvdd_Net ../log/test_reuters ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 0`
    - `--normal_class`可取`0-6`
    - auc: 0/93.93%, 1/90.11%, 2/89.74%, 3/97.93%, 4/82.35%, 5/92.64%, 6/97.62%
- CVDD+newsgroups20: `python3 main.py newsgroups20 cvdd_Net ../log/test_newsgroups20 ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
    - `--normal_class`可取`0-5`
    - auc: 0/74.22%, 1/59.90%, 2/58.05%, 3/75.24%, 4/71.05%, 5/77.78%
- CVDD+imdb: `python3 main.py imdb cvdd_Net ../log/test_imdb ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 0`
    - `--normal_class`可取`0-1`
    - auc: 0/46.44%, 1/56.27%