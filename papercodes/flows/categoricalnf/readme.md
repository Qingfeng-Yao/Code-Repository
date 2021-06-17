## 参考资料
- 2021 | ICLR | Categorical Normalizing Flows via Continuous Transformations | Phillip Lippe and Efstratios Gavves
- [phlippe/CategoricalNF](https://github.com/phlippe/CategoricalNF)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 语言建模实验`experiments/language_modeling`
	- `CNF+PTB`: `python3 train.py --dataset penntreebank --max_iterations 25000 --eval_freq 500 --max_seq_len 288 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 1 --coupling_num_mixtures 51 --coupling_dropout 0.3 --coupling_input_dropout 0.1 --optimizer 4 --learning_rate 7.5e-4 --cluster`
	- `CNF+text8`: `python3 train.py --dataset text8 --max_iterations 100000 --max_seq_len 256 --batch_size 128 --encoding_dim 3 --coupling_hidden_layers 2 --coupling_num_mixtures 27 --coupling_dropout 0.0 --coupling_input_dropout 0.05 --optimizer 4 --learning_rate 7.5e-4 --cluster`
		- 出现问题`FileNotFoundError: [Errno 2] No such file or directory: 'text8/vocabulary.json'`，需要修改函数get_vocabulary的参数root为"data/"，然后在函数内增加创建词汇的函数
		- 还需要在数据集内定义参数PRETRAIN_DIR
	- 若要运行LSTM，则加上参数`--use_rnn`
		- 运行`LSTM`模型时，将`time_dp_rate`改成`input_dp_rate`
		- 运行`LSTM`模型时，出现问题`'TaskLanguageModeling' object has no attribute '_class_loss'`，改成`_calc_loss`
		- 运行`LSTM`模型时，出现问题`AttributeError: 'Tensor' object has no attribute 'zeros_like'`，改成`torch.zeros_like`
		- 目前LSTM的bpd无法复现

## 数据集统计
|  | PTB | Text8 |
| :----: | :----: | :----: |
| vocab | 51 | 27 | 
| train | 41801 | 5624983 | 
| valid | 3360 | 19529 | 
| test | 3739 | 19529 | 

## 运行结果
|  | PTB | Text8 |
| :----: | :----: | :----: |
| LSTM | - | - | 
| CNF | 1.26bpd | 1.45bpd |  