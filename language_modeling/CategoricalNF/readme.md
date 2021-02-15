## 数据集
- penntreebank: 
    - `from torchnlp.datasets import penn_treebank_dataset`
- text8: 
    - `http://mattmahoney.net/dc/text8.zip`
- wikitext103: 
    - `from torchtext.datasets import WikiText103`
    - `GloVe-840B-300`

|  | penntreebank | text8 | wikitext103 | 
| :----: | :----: | :----: | :----: |
| K | 51 | 27 | 10002 | 
| level | character | character | word | 
| max sequence length | 288 | 256 | 256 | 
| num of train sentences | 41801 | 5624983 | 4129069 | 
| num of valid sentences | 3360 | 19529 | 8694 | 
| num of test sentences | 3739 | 19529 | 9811 | 
| num of train tokens | 4751151 | 90000000 | 103227021 | 
| num of valid tokens | 375882 | 5000000 | 217646 | 
| num of test tokens | 416353 | 5000000 | 245569 | 


## 实现模型
- LSTM
- categorical flow model

## 指标
- NLL(负对数似然)，单位bpd(bits per dimension)

## 具体实现命令
- 运行程序前需要提前下载好数据集，存放在目录data下
- 目前资源暂不支持实现wikitext103
- LSTM模型实现
    - penntreebank: `python3 main.py --use_rnn --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4` 
    - text8: `python3 main.py --use_rnn --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`  
    - wikitext103: `python3 main.py --use_rnn --dataset wikitext --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 10 --coupling_hidden_layers 2 --coupling_num_mixtures 64 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`
- categorical flow model实现(去掉参数`--use_rnn`)
    - 只使用一个编码层+一个actnorm层+一个耦合层/AutoregressiveMixtureCDFCoupling
    - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4` 
    - text8: `python3 main.py --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4` 
    - wikitext103: `python3 main.py --dataset wikitext --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 10 --coupling_hidden_layers 2 --coupling_num_mixtures 64 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`
- 以上categorical flow model实现的编码层是线性类编码，另有变分去量化以及变分类编码
    - 变分去量化(添加噪声)(添加参数`--encoding_dequantization`):
        - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --encoding_dequantization`  
    - 变分类编码(使用decoder，而线性类编码不使用decoder)(添加参数`--encoding_use_decoder --encoding_variational`)
        - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --encoding_use_decoder --encoding_variational`  

## 实验结果
以下结果只运行一次
|  | penntreebank | text8 |
| :----: | :----: | :----: | 
| LSTM | 1.41pbd | 1.44pbd | 
| categorical flow model | 1.39pbd | 1.46pbd | 

categorical flow model
|  | 线性类编码 | 变分去量化 | 变分类编码 | 
| :----: | :----: | :----: | :----: | 
| penntreebank | 1.39pbd | 2.38pbd | 1.39pbd |