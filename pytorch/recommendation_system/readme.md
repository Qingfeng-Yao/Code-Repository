### CTR
- 环境配置
    - `python3.6`; `news_recommendation/requirements.txt`
        - 可设置conda环境(newspt)
- 数据集
    - MIND(small)
        - news.tsv: 以'\t'隔开，第一个为newsid，第二个为category one，第三个为category two，第四个为title(word list)
            - 得到一个news字典，key为newsid，value为一个元组，包含category one, category two和title的word list；之后再得到一个newsid2index的字典newsidenx；针对newsidenx字典中的index顺序，可分别得到word_dict, categ_dict和news_features列表(以title和category作为news feature，其中title被title_size限制)
        - behaviors.tsv: 以'\t'隔开，其中第四个为news id序列(历史点击序列)，第五个为候选news序列；每一个正样本匹配若干个负样本，得到一个多分类问题
            - 历史点击序列被his_size限制，训练集中针对每一个用户分别统计正负候选news，然后一个训练样例由随机选取negnums个负候选news和一个正候选news构成的列表，样例对应的标签为正候选news的索引；此外该训练样例还包括用户历史点击序列和实际的点击序列长度(用于mask)
            - 对于测试集，历史点击序列同样被his_size限制；针对每一个用户，统计所有的正负候选news及其1/0标签；每一个用户对应一个测试样例，包括该用户的历史点序列、实际点击序列长度、正负候选news列表和对应的1/0标签列表
        - [数据链接](https://msnews.github.io/index.html)
        - 使用预训练向量Glove
        - 数据集的相关统计信息: `# users: 94057`, `# news: 65238`, `# ave words in news title: 11.77`, `# impressions: 230117`, `# clicks: 347727`, `# train samples: 236344`, `# test samples: 73152`
            - (bert): `# ave words in news title: 13.87`
            - 一个impression表示一条行为记录，一个click表示一条行为记录中一个候选正样本
            - 新闻ctr值的统计: 点击量/曝光量，前者统计每一条记录中的候选正样本，后者为上述的impressions值
    - heybox
        - 原始数据包括用户日志数据(形如xxxx-xx-xx-contact.csv)、帖子数据(形如links_x.json)；以MIND的方式处理数据
        - 中文分词工具jieba，随机初始化词向量 
        - news.tsv: 以'\t'隔开，第一个为帖子id，第二个为用户id，第三个为主题，第四个为标题，第五个为正文
            - 得到一个news字典，key为帖子id，value为一个元组，包含用户id、主题和标题的word list；之后再得到一个帖子id2index的字典newsidenx；针对newsidenx字典中的index顺序，可分别得到word_dict, post_user_dict, categ_dict和news_features列表(以标题、用户id、主题作为post feature，其中标题被title_size限制)
        - behaviors.tsv: 以'\t'隔开，第一个为索引值，第二个为用户id，第三个为时间戳，第四个为历史点击序列，第五个为候选样本序列(包括多个负样本和一个正样本)
            - 历史点击序列被his_size限制，训练集中针对每一个用户分别统计正负候选帖子，然后一个训练样例由随机选取negnums个负候选帖子和一个正候选帖子构成的列表，样例对应的标签为正候选帖子的索引；此外该训练样例还包括用户历史点击序列和实际的点击序列长度(用于mask)
            - 对于测试集，历史点击序列同样被his_size限制；针对每一个用户，统计所有的正负候选样本及其1/0标签；每一个用户对应一个测试样例，包括该用户的历史点序列、实际点击序列长度、正负候选帖子列表和对应的1/0标签列表
        - 数据集的相关统计信息: `# users: 230343`, `# posts: 737507`, `# ave words in post title: 5.91`, `# impressions: 5332771`, `# clicks: 5332771`、`# train samples: 2919193`, `# test samples: 2413578`
            - (bert): `# ave tokens in post title: 12.66`
            - 帖子ctr统计同新闻
- 模型
    - NRMS
    - (+bert)
    - (+mimn)
    - (+co_attention)
    - +din
        - +cross_atten
        - +cross_atten_deep
    - +ctr
        - +din
        - +score_gate
    - 特征融合
        - 标量值融合
            - +score_add
        - 池化
            - +add
            - +mean
            - +max
            - +atten
        - 连接
            - +dnn
            - +moe
                - +bias
            - +mvke
- 指标
    - AUC
    - MRR: Mean Reciprocal Rank(把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均)
    - nDCG(@5 or @10): Normalized Discounted Cumulative Gain(先计算增益，再计算折算因子，最后求和归一化)
- 相关执行命令
    - `MIND`:
        - `NRMS`: python3 main.py --pretrained_embeddings glove; `auc: 65.82, mrr: 30.76, ndcg5: 33.76, ndcg10: 40.23`
        - `NRMS+din`: python3 main.py --pretrained_embeddings glove --din; `auc: 65.98, mrr: 31.27, ndcg5: 34.38, ndcg10: 40.69`
        - NRMS+din+cross_atten: python3 main.py --pretrained_embeddings glove --cross_atten; auc: 65.96, mrr: 30.94, ndcg5: 33.93, ndcg10: 40.39
        - NRMS+din+cross_atten_deep: python3 main.py --pretrained_embeddings glove --cross_atten --cross_atten_deep; auc: 65.71, mrr: 30.77, ndcg5: 33.68, ndcg10: 40.18
        - NRMS+ctr: python3 main.py --pretrained_embeddings glove --use_ctr; auc: 65.79, mrr: 30.75, ndcg5: 33.76, ndcg10: 40.22
        - `NRMS+ctr+score_gate`: python3 main.py --pretrained_embeddings glove --use_ctr --score_gate; `auc: 66.09, mrr: 30.97, ndcg5: 34.03, ndcg10: 40.48`
        - NRMS+ctr+din: python3 main.py --pretrained_embeddings glove --use_ctr --din; auc: 64.84, mrr: 30.44, ndcg5: 33.11, ndcg10: 39.77

        - `NRMS+add`: python3 main.py --pretrained_embeddings glove --din --use_ctr --add_op; `auc: 66.31, mrr: 31.30, ndcg5: 34.41, ndcg10: 40.80`
        - `NRMS+mean`: python3 main.py --pretrained_embeddings glove --din --use_ctr --mean_op; `auc: 66.25, mrr: 31.30, ndcg5: 34.44, ndcg10: 40.79`
        - `NRMS+max`: python3 main.py --pretrained_embeddings glove --din --use_ctr --max_op; auc: `65.95, mrr: 31.13, ndcg5: 34.21, ndcg10: 40.61`
        - `NRMS+atten`: python3 main.py --pretrained_embeddings glove --din --use_ctr --atten_op; `auc: 66.26, mrr: 31.31, ndcg5: 34.40, ndcg10: 40.82`

        - `NRMS+score_add`: python3 main.py --pretrained_embeddings glove --din --use_ctr --score_add; `auc: 65.99, mrr: 31.04, ndcg5: 34.13, ndcg10: 40.52`

        - `NRMS+dnn`: python3 main.py --pretrained_embeddings glove --din --use_ctr --dnn; `auc: 66.21, mrr: 31.43, ndcg5: 34.56, ndcg10: 40.93`
        - `NRMS+moe`: python3 main.py --pretrained_embeddings glove --din --use_ctr --moe; `auc: 66.68, mrr: 31.60, ndcg5: 34.70, ndcg10: 41.08`
        - NRMS+bias: python3 main.py --pretrained_embeddings glove --din --use_ctr --moe --bias; auc: 66.19, mrr: 31.32, ndcg5: 34.37, ndcg10: 40.75
        - NRMS+mvke: python3 main.py --pretrained_embeddings glove --din --use_ctr --mvke; auc: 65.70, mrr: 31.08, ndcg5: 34.05, ndcg10: 40.47

        - NRMS+mimn: python3 main.py --pretrained_embeddings glove --mimn; auc: 65.24, mrr: 30.71, ndcg5: 33.79, ndcg10: 40.14
        - NRMS+bert: python3 main.py --word_embed_size 768 --pretrained_embeddings bert --batch_size 4;
    - `heybox`:
        - `NRMS`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 5e-5 --epochs 10; `auc: 65.99, mrr: 46.51, ndcg5: 52.46, ndcg10: 59.26`
        - `NRMS+din`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 5e-5 --epochs 10 --din; `auc: 66.19, mrr: 46.77, ndcg5: 52.71, ndcg10: 59.45`
        - `NRMS+ctr+score_gate`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 5e-5 --epochs 10 --use_ctr --score_gate; `auc: 66.16, mrr: 46.58, ndcg5: 52.60, ndcg10: 59.31`
        - `NRMS+moe`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 5e-5 --epochs 10 --din --use_ctr --moe; `auc: 66.11, mrr: 46.73, ndcg5: 52.66, ndcg10: 59.42`
        

- 参考论文
    - 2019 | EMNLP | Neural News Recommendation with Multi-Head Self-Attention | Microsoft
    - 2019 | KDD | Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction | Alibaba
    - 2021 | SIGIR | Empowering News Recommendation with Pre-trained Language Models | Microsoft
    - 2021 | SIGIR | Personalized News Recommendation with Knowledge-aware Interactive Matching | Microsoft
    - 2021 | ACL | PP-Rec: News Recommendation with Personalized User Interest and Time-aware News Popularity | Microsoft
- 参考资料
    - [wuch15/EMNLP2019-NRMS](https://github.com/wuch15/EMNLP2019-NRMS)
    - [wuch15/PLM4NewsRec](https://github.com/wuch15/PLM4NewsRec)
    - [UIC-Paper/MIMN](https://github.com/UIC-Paper/MIMN)
    

    
        