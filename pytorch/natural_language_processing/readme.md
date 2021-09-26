## �ο�����
### text modeling
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)
- [TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows)
- [google/edward2](https://github.com/google/edward2)

### text anomaly detection
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)

## �ο�����
### text modeling
- 2019 | NIPS | Discrete Flows: Invertible Generative Models of Discrete Data | Dustin Tran et al.(Google)
- 2021 | ICLR | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves

### text anomaly detection
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | Lukas Ruff et al.

## ��������
### text modeling
- `python3.6`; `text_modeling/requirements.txt`
    - ������conda����(textgmpy)

### text anomaly detection
- `python3.7`; `text_anomaly_detection/requirements.txt`
    - ����`spaCy en`��: `python3 -m spacy download en`;���ܻ���Ϊ`timed out`�����޷����ص����⣬�ɶ�γ��ԣ�ʵ�ڲ���ֻ�������Ѿ����غõ�enĿ¼������`python3.7/site-packages/spacy/data`Ŀ¼��
    - ������conda����(textanopy)

## ����
### text(language) modeling
#### ���ݼ�
- Penn Treebank
- text8
- ִ��ÿ�������ļ��е�main�����ֽ���Ӧ���ݼ����ص�`data`�ļ����£�`Penn Treebank`�������г���Ϊ(0,288)���ʻ���Ϊ51��`text8`�������г���Ϊ256���ʻ���Ϊ27�������ַ�����
- `Penn Treebank`ѵ��������������41801�����ַ���4751151����֤������������3360�����ַ���375882�����Լ�����������3739�����ַ���416353 
- `text8`ѵ��������������5624983�����ַ���90000000����֤������������19529�����ַ���5000000�����Լ�����������19529�����ַ���5000000

#### ģ��
- LSTM
    - Ƕ���: ����ɢ������
    - ʱ��Ƕ���: ��ԭʼǶ��������Ƕ��ά��
    - LSTM����
    - �����: ���ά��Ϊ�ʻ���С�����Բ�
        - LSTM���羭�����õ��ĸ���Ϊ��һ��token��Ԥ�����
        - ��һ��token��Ԥ������ɳ�ʼstate�����������õ�
- Categorical Normalizing Flow
    - �����: ������ֲ��в������ݲ����б任��������ɢ���ݵ�Ƕ����Ʊ任�Ĳ���
    - �����: �任�Ĳ���Ϊ��ѧϰ����
    - �Իع�����ϲ�
- Discrete Flows: 
    - Discrete autoregressive flows
    - Discrete bipartite flows
    - �����������one-hot���룬Ȼ��DAFʹ��MADE�任��DBFʹ�ú�ȱ任

#### ���ִ������
- ͳһʵ������: use_multi_gpu/False, restart/False, checkpoint_path/None, load_config/False, no_model_checkpoints/False, only_eval/False, cluster/True, debug/False, clean_up/False
- ָ��: ���Լ��ϵĸ�������Ȼ(test bpd), ����һ����������Ҫ��ʱ��(generate, ��λs, PTB��Ӧ���г���Ϊ288, text8��Ӧ���г���Ϊ256)
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
#### ���ݼ�
- �ֱ�ͳ��ÿ�����ݼ���ÿ����Ϊ��������ʱ��ѵ����(ֻ������������)�����Լ�(ͬʱ�����������ݺ��쳣����)�Լ�Ĭ�������µ�����г��ȣ���ֵ������ÿ�������������
- �ʻ���С(������SpacyEncoder�е�min_occurrences�����ʻ���С��Ĭ��Ϊ3)
- Reuters-21578
    - ��7��: earn(2840/1083/1097/494), acq(1596/696/1484/502), crude(253/121/2059/484), trade(250/76/2104/524), money-fx(222/87/2093/502), interest(191/81/2099/484), ship(108/36/2144/484)
    - ���Լ����ܴ�С����2180
- 20 Newsgroups
    - ��6��: comp(2857/1909/5390/7337), rec(2301/1524/5775/7337), sci(2311/1520/5779/7337), misc(577/382/6917/7337), pol(1531/1025/6274/7337), rel(1419/939/6360/7337)
    - ���Լ����ܴ�С����7299
- IMDB
    - ��2��: pos(12500/12500/12500/1390), neg(12500/12500/12500/1132)
    - ���Լ����ܴ�С����25000
- �ȴ���Ŀ¼`data`��Ȼ�����`src`����`Reuters-21578`��ִ��`import nltk;nltk.download('reuters', download_dir='../data');nltk.download('stopwords', download_dir='../data');nltk.download('punkt', download_dir='../data')`�������������ݼ��Լ���Ӧ��Ԥѵ�����������Զ�����
- �����ı����������µ�Ԥ����: Сд���Ƴ���㡢ǰ��ո�ͣ�ôʺͳ���С��3�Ĵ�
- ÿ���������ֵ�`{'text': text,'label': label}`���ɣ���ǩ������Ƿ��������Ϊ0(����)��1(�쳣)�����ж�`Reuters-21578`��ֻ�е���ǩ����������
- ѵ���������������������Լ�������������

#### ģ��
- CVDD
    - ������ע��������ѧϰ���еı�ʾ��ͬʱ������ͬ��״�Ŀ�ѧϰ�������������쳣��������Ϊ��������������������֮������Ҿ���
- CNF 
    - ����CNF�������е���ɢ��Ȼ�����Ը�������ȻΪ�쳣����
- EmbeddingNF
    - �����׼�����������е�Ƕ����Ȼ�����Ը�������ȻΪ�쳣����

#### ���ִ������
- ������������ǰ��Ҫ�ȴ�����־Ŀ¼`log`
- CNF�޷�ʹ��Ԥѵ����Ƕ�룬��Ҫʹ�ø�����Դ棬��ʱֻ�ܴ���reuters
- CVDD+reuters: `python3 main.py reuters cvdd_Net ../log ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --max_seq_len 550 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 0`
    - `--normal_class`��ȡ`0-6`
    - auc: 0/93.88%, 1/90.14%, 2/89.63%, 3/98.18%, 4/77.73%, 5/92.86%, 6/97.64%
- CNF+reuters: `python3 main.py reuters CNF ../log ../data --seed 1 --clean_txt  --num_dimensions 3 --coupling_hidden_layers 1 --coupling_num_flows 1 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --use_length_prior --use_time_embed --max_seq_len 550 --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
    - `--normal_class`��ȡ`0-6`
    - auc: 0/%, 1/%, 2/%, 3/%, 4/%, 5/%, 6/



- EmbeddingNF+reuters: `python3 main.py reuters EmbeddingNF ../log ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --coupling_hidden_layers 1 --coupling_num_flows 1 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --use_length_prior --use_time_embed --max_seq_len 550 --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
    - `--normal_class`��ȡ`0-6`
    - auc: 0/%, 1/%, 2/%, 3/%, 4/%, 5/%, 6/%
- CVDD+newsgroups20: `python3 main.py newsgroups20 cvdd_Net ../log ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --max_seq_len 7337 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
    - `--normal_class`��ȡ`0-5`
    - auc: 0/74.30%, 1/60.02%, 2/58.14%, 3/75.64%, 4/71.06%, 5/77.88%




- EmbeddingNF+newsgroups20: `python3 main.py newsgroups20 EmbeddingNF ../log ../data --seed 1 --clean_txt --embedding_size 3 --coupling_hidden_layers 1 --coupling_num_flows 1 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --max_seq_len 7337 --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0`
    - `--normal_class`��ȡ`0-5`
    - auc: 0/80.64%, 1/59.90%, 2/58.05%, 3/75.24%, 4/71.05%, 5/77.78%



- CVDD+imdb: `python3 main.py imdb cvdd_Net ../log ../data --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --max_seq_len 1400 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 0`
    - `--normal_class`��ȡ`0-1`
    - auc: 0/46.39%, 1/56.26%