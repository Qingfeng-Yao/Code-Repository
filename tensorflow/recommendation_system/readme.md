## 参考资料
- [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)

## 参考论文
- 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction | Alibaba
- 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Google
- 2020 | Hybrid Interest Modeling for Long-tailed Users | Alibaba
- 2021 | One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction | Alibaba

## 环境配置
- `python3.6`; `requirements.txt`
- `conda install cudatoolkit=10.0`
- `conda install cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/`

## 任务
### CTR
#### 数据集
- amazon(eletronics)
- movielens

#### 模型
- 使用tf.estimator.Estimator，参数model_fn指明模型(tf.estimator.EstimatorSpec)，参数config指明配置(tf.estimator.RunConfig)
- 模型函数的输入为: features, labels, mode, params，分别是批特征、批标签、模式和模型参数，其中模式由tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT来定义
- 模型函数的输出为tf.estimator.EstimatorSpec
- 模型训练: 分别定义tf.estimator.TrainSpec和tf.estimator.EvalSpec，然后和模型一起传入到tf.estimator.train_and_evaluate中
- 模型预测: 调用estimator.predict方法
- 具体模型(指标AUC)
    - DIN: 0.7506
    - MMOE: 0.7566
    - BIAS: 0.7624
    - `STAR: 0.7687`
    - UserLoss: 0.7608
    - UBC: 
        