## 任务
- language modeling(连续性假设)
    - 分布内数据集：penn | pennchar
    - 分布外数据集：English Web Treebank: Yahoo! Answers | emails | newsgroups | product reviews | weblogs
    - 异常暴露数据集：wikitext-2
    - 备注：全部使用分布内数据集的字典，未见token用`<unk>`表示

## 参考资料
- [hendrycks/outlier-exposure](https://github.com/hendrycks/outlier-exposure)
    - wikitextchar处理以及OOD语料库处理
    - OE损失：所有词汇的平均概率或加权概率
    - ASGD优化器及模型参数
    - 异常分数定义：与目标的差距或全词汇概率或单一词汇概率
    - 使用：
        - `python3 main_rnn.py --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 50 --bptt 150 --resume_ood output/LSTM-penn-FalseOE/model.pt`
        - `python3 main_rnn.py --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 50 --bptt 150 --resume_ood output/LSTM-pennchar-FalseOE/model.pt --wikitext_char --dataset pennchar`
        - `python3 main_rnn.py --batch_size 20 --use_OE --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 5 --bptt 150 --resume_oe output/LSTM-penn-FalseOE/model.pt --resume_ood output/LSTM-penn-TrueOE/model.pt`