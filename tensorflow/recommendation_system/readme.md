### CTR
- 环境配置
    - `python3.6`; `CTR/requirements.txt`
        - `conda install cudatoolkit=10.0`
        - `conda install cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/`
        - 可设置conda环境(ctrtf)
- 数据集
    - amazon(eletronics)
    - movielens
    - heybox
    - 具体的数据集处理过程及相关统计信息可参见`CTR/data`
- 模型
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
- 相关执行命令
    - `DIN`:
        -  `amazon`: 
- 参考论文
    - 2017 | ICLR | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Google
    - 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction | Alibaba
    - 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Google
    - 2020 | Hybrid Interest Modeling for Long-tailed Users | Alibaba
    - 2021 | One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction | Alibaba
    - 2021 | DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning | Google
    - 2021 | Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling | Tencent
- 参考资料
    - [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)
    - [sparse gate](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py)
    - [DSelect k](https://github.com/google-research/google-research/tree/master/dselect_k_moe)
    

    
        