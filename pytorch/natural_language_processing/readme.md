## 参考资料
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)

## 参考论文
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.
- 2019 | NIPS | Discrete Flows: Invertible Generative Models of Discrete Data | Dustin Tran et al.
- 2021 | ICLR | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves

## 环境配置
- `python3.6`; `requirements.txt`

## 任务
### text(language) modeling
#### 数据集
- Penn Treebank
- text8
- 执行每个数据文件中的main函数现将对应数据集下载到`data`文件夹下，`Penn Treebank`限制序列长度为(0,288)，`text8`限制序列长度为256，均是字符级别
- `Penn Treebank`训练集中序列总数41801，总字符数4751151；验证集中序列总数3360，总字符数375882；测试集中序列总数3739，总字符数416353 
- `text8`训练集中序列总数5624983，总字符数90000000；验证集中序列总数19529，总字符数5000000；测试集中序列总数19529，总字符数5000000

#### 模型
- LSTM
- Categorical Normalizing Flow

#### 相关执行命令
- PTB+CNF: `python3 train.py --dataset penntreebank --model_name CNF --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.26
- PTB+RNN: `python3 train.py --dataset penntreebank --model_name RNN --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.273 
- text8+CNF: `python3 train.py --dataset text8 --model_name CNF --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.448
- text8+RNN: `python3 train.py --dataset text8 --model_name RNN --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.435 

### text anomaly detection
#### 数据集


#### 模型