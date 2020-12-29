## 任务
- language modeling(连续性假设)
    - 分布内数据集：penn | pennchar
    - 分布外数据集：English Web Treebank: Yahoo! Answers | emails | newsgroups | product reviews | weblogs
    - 异常暴露数据集：wikitext-2
    - 备注：全部使用分布内数据集的字典，未见token用`<unk>`表示
- one-class classification
    - Reuters-21578/7类、20 Newsgroups/6类

## 参考文献
- 2019 | ICLR | Deep Anomaly Detection with Outlier Exposure | Dan Hendrycks et al.
- 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text | LukasRuff et al.

## 参考代码
- [hendrycks/outlier-exposure](https://github.com/hendrycks/outlier-exposure)
    - wikitextchar处理以及OOD语料库处理
    - OE损失：所有词汇的平均概率或加权概率
    - ASGD优化器及模型参数
    - 异常分数定义：与目标的差距或全词汇概率或单一词汇概率
    - 测试指标：AUROC AUPR FPR 
    - 使用：
        - `python3 main_rnn.py --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 50 --bptt 150 --resume_ood output/LSTM-penn-FalseOE/model.pt`
        - `python3 main_rnn.py --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 50 --bptt 150 --resume_ood output/LSTM-pennchar-FalseOE/model.pt --wikitext_char --dataset pennchar`
        - `python3 main_rnn.py --batch_size 20 --use_OE --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --dropouti 0.4 --epochs 5 --bptt 150 --resume_oe output/LSTM-penn-FalseOE/model.pt --resume_ood output/LSTM-penn-TrueOE/model.pt`
- [lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
    - 因涉及nltk的tokenize和stopwords，所以可以提前下载好corpora/stopwords和tokenizers
    - 没有使用数据集imdb 
    - 数据预处理：不考虑tfidf权重、只使用spacy划分、不考虑eos和sos
    - 不使用bert且预训练嵌入不更新及归一化
    - 使用：
        - `python3 main_cvdd.py --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 80 --normal_class 0`
        - `python3 main_cvdd.py --dataset newsgroup --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 80 --normal_class 0`
