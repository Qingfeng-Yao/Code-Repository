## 任务
- 语言建模

## 数据集
- PTB(Penn Treebank): character/word
- enwik8(Hutter Prize dataset): character
- WikiText-2(WT2): word
- WikiText-103(WT103): word
- 格式：txt或来自torchnlp.datasets

## 模型
- RNN：LSTM、GRU、QRNN
- 流模型

## 指标
- NLL(负对数似然)，单位bpc/bpw，即bits per character/bits per word

## 特点
- 提升GPU利用率

## 参考文献
- 2018 | ICLR | Regularizing and Optimizing LSTM Language Models | Stephen Merity et al.
- 2018 | An Analysis of Neural Language Modeling at Multiple Scales | Stephen Merity et al.
- 2019 | NIPS | Discrete flows: Invertible generative models of discrete data | Dustin Tran et al.
    - discrete flow model
- 2019 | ICML | Latent Normalizing Flows for Discrete Sequences | Zachary M. Ziegler and Alexander M. Rush 
    - latent flow model
- 2020 | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves
    - categorical flow model

## 参考代码
- [salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)
    - 未使用QRNN
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
    - flow+lstm
        - LSTM: `python3 main_categorical_flows.py --use_rnn --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4`或者`python3 main_categorical_flows.py --use_rnn --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`或者`python3 main_categorical_flows.py --use_rnn --dataset wikitext --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 10 --coupling_hidden_layers 2 --coupling_num_mixtures 64 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`
    - 运行程序前需要先下载好数据集，在nlpdata文件下
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)
    - 离散高斯混合
    - 采样(reverse)：scale出现0，无法进行除法
    - 还未考虑二部流
- [harvardnlp/TextFlow](https://github.com/harvardnlp/TextFlow)