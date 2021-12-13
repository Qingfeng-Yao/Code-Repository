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
    - +ctr
    - +din
    - 特征物理融合
        - 池化
        - 连接
    - (+co_attention)
- 指标
    - AUC
    - MRR: Mean Reciprocal Rank(把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均)
    - nDCG(@5 or @10): Normalized Discounted Cumulative Gain(先计算增益，再计算折算因子，最后求和归一化)
- 相关执行命令
    - `MIND`: 均考虑score_gate，即最后计算分数时同时考虑匹配分数和候选新闻的ctr值
        - `NRMS`: python3 main.py --pretrained_embeddings glove --score_gate; `auc: 66.25, mrr: 31.67, ndcg5: 34.77, ndcg10: 41.09`
        - `NRMS+ctr`: python3 main.py --pretrained_embeddings glove --use_ctr --score_gate; `auc: 66.45, mrr: 31.83, ndcg5: 34.95, ndcg10: 41.25`
        - `NRMS+din`: python3 main.py --pretrained_embeddings glove --din --score_gate; `auc: 66.67, mrr: 31.84, ndcg5: 34.92, ndcg10: 41.36`
        - `NRMS+atten`: python3 main.py --pretrained_embeddings glove --din --use_ctr --atten_op; `auc: 66.66, mrr: 31.87, ndcg5: 35.05, ndcg10: 41.42`
    - `heybox`: 5 epoch
        - `NRMS`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 1e-5 --score_gate; `auc: 64.88, mrr: 45.77, ndcg5: 51.44, ndcg10: 58.65`
        - `NRMS+ctr`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 1e-5 --use_ctr --score_gate; `auc: 64.91, mrr: 45.72, ndcg5: 51.45, ndcg10: 58.62`
        - `NRMS+din`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 1e-5 --din --score_gate; `auc: 65.59, mrr: 46.17, ndcg5: 52.06, ndcg10: 58.99`
        - `NRMS+atten`: python3 main.py --dataset heybox --title_size 10 --his_size 50 --neg_number 10 --batch_size 512 --lr 1e-5 --din --use_ctr --atten_op; `auc: 65.27, mrr: 46.23, ndcg5: 51.89, ndcg10: 59.01`
        

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
    

    
        