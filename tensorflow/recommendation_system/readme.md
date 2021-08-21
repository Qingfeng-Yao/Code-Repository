## 参考资料
- [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)

## 参考论文
- 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction

## 环境配置
- `python3.6`; `requirements.txt`
- `conda install cudatoolkit=10.0`
- `conda install cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/`

## 任务
### CTR
- amazon(eletronics)
    - reviews_Electronics_5.json
        - 每一行形如`{'reviewerID': 'A2JXAZZI9PHK9Z', 'asin': '0594451647', 'reviewerName': 'Billy G. Noland "Bill Noland"', 'helpful': [3, 3], 'reviewText': 'I am using this with a Nook HD+. It works as described. The HD picture on my Samsung 52&#34; TV is excellent.', 'overall': 5.0, 'summary': 'HDMI Nook adapter cable', 'unixReviewTime': 1388707200, 'reviewTime': '01 3, 2014'}`，共1689188行，得到reviews.pkl
        - 每一行只保留`['reviewerID', 'asin', 'unixReviewTime']`，并将'reviewerID', 'asin'对应对值转为索引
    - meta_Electronics.json
        - 每一行形如`{'asin': '0558835155', 'description': 'Use these high quality headphones for internet chatting and enjoy the comfort and ease of the headphones with the microphone and in-line volume control.Works with: Skype msn AIM YAHOO! Windows Live', 'title': 'Polaroid Pbm2200 PC / Gaming Stereo Headphones With Microphone &amp; In-line Volume', 'price': 13.95, 'imUrl': 'http://ecx.images-amazon.com/images/I/21rEirndRLL.jpg', 'categories': [['Electronics', 'Accessories & Supplies', 'Audio & Video Accessories', 'Headphones']]}`，共498196行，对asin去重得到meta.pkl，共63001行
        - 每一行只保留`['asin', 'categories']`，其中categories只取最后一个类，并将'asin', 'categories'对应对值转为索引
    - remap.pkl: 由4部分组成，分别是reviews_df、cate_list(每个item对应类组成的列表)和两个元组(一个由user_count, item_count, cate_count, example_count组成，另一个由asin_key, cate_key, revi_key三个字典组成，三个字典分别对应itemid、类id和用户id)
        - 用户数192403，item数63001，类数801，样本数1689188
    - 生成和正样本一样数目的负样本，并按用户进行分组；每个训练集和测试集中的每一个样本是一个元组，包含用户id、历史序列、itemid以及是否评论(0或1)；dataset.pkl由train_set和test_set组成，前者共有2608764，后者共有384806；假设序列长度为n，任务是利用前k个物品预测第k+1个；训练集中的k取1,2,...,n-2，测试集k取n-1
    - 最后生成amazon_train.tfrecords和amazon_valid.tfrecords，涉及特征`[reviewer_id, hist_item_list, hist_category_list, hist_length, item, item_category, target]`
- 模型
    - 使用tf.estimator.Estimator，参数model_fn指明模型(tf.estimator.EstimatorSpec)，参数config指明配置(tf.estimator.RunConfig)
    - 模型函数的输入为: features, labels, mode, params，分别是批特征、批标签、模式和模型参数，其中模式由tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT来定义
    - 模型函数的输出为tf.estimator.EstimatorSpec
    - 模型训练: 分别定义tf.estimator.TrainSpec和tf.estimator.EvalSpec，然后和模型一起传入到tf.estimator.train_and_evaluate中
    - 模型预测: 调用estimator.predict方法
    - 具体模型(指标AUC, global step)
        - DIN: 0.6851(85400)   评估(200step)
        - DIN: 0.6823(79700)   评估(100step/全数据集)
        - DIN: 0.7242(83100)   训练(小数据集)
        - MMOE: 0.7312(80050)
        - BIAS: 0.7497(86801)
        - UBC: 0.7307(87400)
        - STAR: 0.7604(102250)