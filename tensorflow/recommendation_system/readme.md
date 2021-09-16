## 参考资料
### CTR
- [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)
- [sparse gate](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py)
- [DSelect k](https://github.com/google-research/google-research/tree/master/dselect_k_moe)

## 参考论文
### CTR
- 2017 | ICLR | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Google
- 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction | Alibaba
- 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Google
- 2020 | Hybrid Interest Modeling for Long-tailed Users | Alibaba
- 2021 | One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction | Alibaba
- 2021 | DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning | Google
- 2021 | Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling | Tencent

## 环境配置
### CTR
- `python3.6`; `CTR/requirements.txt`
    - `conda install cudatoolkit=10.0`
    - `conda install cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/`
    - 可设置conda环境(ctrtf)

## 任务
### CTR
#### 数据集
- amazon(eletronics)
- movielens
- 具体的数据集处理过程及相关统计信息可参见`CTR/data`

#### 模型
- 使用tf.estimator.Estimator，参数model_fn指明模型(tf.estimator.EstimatorSpec)，参数config指明配置(tf.estimator.RunConfig)
- 模型函数的输入为: features, labels, mode, params，分别是批特征、批标签、模式和模型参数，其中模式由tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT来定义
- 模型函数的输出为tf.estimator.EstimatorSpec
- 模型训练: 分别定义tf.estimator.TrainSpec和tf.estimator.EvalSpec，然后和模型一起传入到tf.estimator.train_and_evaluate中
- 模型预测: 调用estimator.predict方法
- 具体模型(指标AUC)
    - DIN: 0.7370/0.9137
    - MOE: 0.7800/0.9440
    - Bias: 0.7796/0.9435
    - (UserInput)
    - (UserLoss)
    - UserCluster: 0.7910
        - +cluster loss
    - UserSparseExpert: 0.7843/0.9439
        - (+DSelect k(static))
        - (+DSelect k(per example))
    - UserPerExpert: 0.7857/0.9437
    

    
        