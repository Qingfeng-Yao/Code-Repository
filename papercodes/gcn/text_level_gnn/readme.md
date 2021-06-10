## 参考资料
- 2019 | EMNLP | Text Level Graph Neural Network for Text Classification | Lianzhe Huang et al.
- [Cynwell/Text-Level-GNN](https://github.com/Cynwell/Text-Level-GNN)

## 源码运行
- 环境配置: `python3.7`; `requirements.txt`
- 命令运行: 
    - `python3 train.py --cuda=0 --embedding_size=300 --p=3 --min_freq=2 --max_length=70 --dropout=0 --epoch=300`: 运行前需要获取预训练嵌入，保存在embeddings
- 实验提供的数据集格式: 包括R8和R52
    - r8-train-all-terms.txt: 用于训练的，标签+文本
    - r8-test-all-terms.txt: 用于测试的，标签+文本

## 运行结果
|  | R8 | 
| :----: | :----: |
| Training Accuracy | 0.9749 |
| Validation Accuracy | 0.9289 | 
| Testing Accuracy | 0.9347 | 