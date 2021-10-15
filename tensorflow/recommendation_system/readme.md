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
    - DIN
    - MOE
    - Bias
    - UserPerExpert
- 相关执行命令
    - `heybox`:
        - `DIN`: python3 main.py --model DIN --dataset heybox; `auc: 0.7186`
        - `MOE`: python3 main.py --model MOE --dataset heybox; `auc: 0.7185`
        - `Bias`: python3 main.py --model Bias --dataset heybox; `auc: 0.7245`
        - `UserPerExpert`: python3 main.py --model UserPerExpert --dataset heybox; `auc: 0.7262`
    - `amazon`    
        - `DIN`: python3 main.py --model DIN --dataset amazon; `auc: 0.7090`
    - `movielens`: 
        - `DIN`:python3 main.py --model DIN --dataset movielens; `auc: 0.6519`

- 参考论文
    - 2018 | KDD | Deep Interest Network for Click-Through Rate Prediction | Alibaba
    - 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Google
    - 2021 | Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling | Tencent
- 参考资料
    - [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)
    

    
        