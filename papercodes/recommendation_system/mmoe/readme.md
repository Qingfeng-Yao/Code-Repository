## 参考资料
- 2018 | KDD | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Jiaqi Ma et al.
- [huangjunheng/recommendation_model](https://github.com/huangjunheng/recommendation_model)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 
    - `python3 mmoe.py`
    - `auc_1 = roc_auc_score(t1_target, t1_pred)`遇到问题`ValueError: unknown format is not supported`，主要是由于第一个参数的数据类型，修改: gpu-cpu-numpy-int
- 数据集: UCI census-income，共有40(实验中有42)个特征(实际训练和测试只用了29个特征)
- 两个多任务学习(二分类问题): 将某些特征作为预测目标，在随机的10000个样本上计算任务标签的Pearson关联的绝对值
    - 任务一: 预测收入是否超过$50K，任务二: 预测婚姻状况是否是永不结婚；绝对Pearson关联是0.1768
    - 任务一: 预测教育水平是否至少是大学，任务二: 预测婚姻状况是否是永不结婚；绝对Pearson关联是0.2373
    - 上述两个学习问题，任务一均为主任务，任务二为辅助任务
- 模型
    - 专家网络和塔网络: 两个线性层的叠加
    - 门网络: 一个线性层+softmax
- 指标: AUC

## 数据集统计
| # train | # valid | # test | # valid+test | 
| :----: | :----: | :----: | :----: |
| 199523 | 49881 | 49881 | 99762 | 

## 问题一运行结果
| task1 | task2 | 
| :----: | :----: |
| 0.943 | 0.970 | 