## 数据集
- penntreebank: 
    - character-level
    - 51类
- text8: 
    - character-level
    - 27类
- wikitext: 
    - wikitext-2; wikitext-103
    - word-level
    - 10002类

## 实现模型
- LSTM
- categorical flow model

## 指标
- NLL(负对数似然)，单位bpd(bits per dimension)

## 具体实现命令
- 运行程序前需要提前下载好数据集，存放在目录data下
- LSTM模型实现
    - penntreebank: `python3 main.py --use_rnn --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4`  测试集指标1.41pbd
    - text8: `python3 main.py --use_rnn --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`  测试集指标1.44pbd
- categorical flow model实现(去掉参数`--use_rnn`)
    - 只使用一个编码层+一个actnorm层+一个耦合层
    - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4`  测试集指标1.39pbd
    - text8: `python3 main.py --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4`  测试集指标1.46pbd
- 以上categorical flow model实现的编码层是线性类编码，另有变分去量化以及变分类编码
    - 变分去量化(添加噪声)(添加参数`--encoding_dequantization`):
        - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --encoding_dequantization`  测试集指标2.38pbd
    - 变分类编码(使用decoder，而线性类编码不使用decoder)(添加参数`--encoding_use_decoder --encoding_variational`)
        - penntreebank: `python3 main.py --dataset penntreebank --max_iterations 100000 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --encoding_use_decoder --encoding_variational`  测试集指标1.39pbd

## 实验结果
|  | penntreebank | text8 |
| :----: | :----: | :----: | 
| LSTM | 1.41pbd | 1.44pbd | 
| categorical flow model | 1.39pbd | 1.46pbd | 

categorical flow model
|  | 线性类编码 | 变分去量化 | 变分类编码 | 
| :----: | :----: | :----: | :----: | 
| penntreebank | 1.39pbd | 2.38pbd | 1.39pbd |