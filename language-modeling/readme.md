## 任务
- 语言建模

## 数据集
- PTB(Penn Treebank): character/word
- enwik8(Hutter Prize dataset): character
- WikiText-2(WT2): word
- WikiText-103(WT103): word

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
- 2019 | ICML | Latent Normalizing Flows for Discrete Sequences | Zachary M. Ziegler and Alexander M. Rush 
- 2020 | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves

## 参考代码
- [salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)
    - 未使用QRNN
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)
- [harvardnlp/TextFlow](https://github.com/harvardnlp/TextFlow)