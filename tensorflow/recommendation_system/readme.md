## 参考资料
- [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)
- [sparse gate](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py)
- [DSelect k](https://github.com/google-research/google-research/tree/master/dselect_k_moe)

## 参考论文
- 2017 | ICLR | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Google
- 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction | Alibaba
- 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Google
- 2020 | Hybrid Interest Modeling for Long-tailed Users | Alibaba
- 2021 | One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction | Alibaba
- 2021 | DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning | Google

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
    - DIN: 0.7370/0.9137
    - MOE: 0.7475/0.9285
    - Bias: 0.7577/0.9328
    - (UserInput: 0.7648/0.9329)
    - (UserLoss: 0.7584/0.9314)
    - (Star: 0.7434/0.9291)
    - (UserRecognition: 0.6869)
    - UserCluster: 0.7268/(bz=16)
        - +ClusterLoss: 0.6973
        - +Adversarial: 
    - UserSparseExpert: 0.7411/0.9403
        - +DSelect k(static): 0.7572
        - +DSelect k(per example): 
    - UserPerExpert

    

    
        