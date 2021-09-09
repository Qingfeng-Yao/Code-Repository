## �ο�����
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)

## �ο�����
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.
- 2019 | NIPS | Discrete Flows: Invertible Generative Models of Discrete Data | Dustin Tran et al.
- 2021 | ICLR | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves

## ��������
- `python3.6`; `requirements.txt`

## ����
### text(language) modeling
#### ���ݼ�
- Penn Treebank
- text8
- ִ��ÿ�������ļ��е�main�����ֽ���Ӧ���ݼ����ص�`data`�ļ����£�`Penn Treebank`�������г���Ϊ(0,288)��`text8`�������г���Ϊ256�������ַ�����
- `Penn Treebank`ѵ��������������41801�����ַ���4751151����֤������������3360�����ַ���375882�����Լ�����������3739�����ַ���416353 
- `text8`ѵ��������������5624983�����ַ���90000000����֤������������19529�����ַ���5000000�����Լ�����������19529�����ַ���5000000

#### ģ��
- LSTM
- Categorical Normalizing Flow

#### ���ִ������
- PTB+CNF: `python3 train.py --dataset penntreebank --model_name CNF --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.26
- PTB+RNN: `python3 train.py --dataset penntreebank --model_name RNN --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.273 
- text8+CNF: `python3 train.py --dataset text8 --model_name CNF --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.448
- text8+RNN: `python3 train.py --dataset text8 --model_name RNN --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
    - test bpd: 1.435 

### text anomaly detection
#### ���ݼ�


#### ģ��