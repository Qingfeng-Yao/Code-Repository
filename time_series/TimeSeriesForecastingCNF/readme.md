## 参考资料
- 2021 | ICLR | Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows | Kashif Rasul et al.
- [zalandoresearch/pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)
- [awslabs/gluon-ts](https://github.com/awslabs/gluon-ts)

## 程序运行
- 环境配置：`python3.7`; `requirements.txt`
- 执行文件`Multivariate-Flow-Solar.ipynb`
    - 执行模型：`GRU-Real-NVP`; `GRU-MAF`; `Transformer-MAF`

## solar_nips执行结果
|  | GRU-Real-NVP | GRU-MAF | Transformer-MAF | 
| :----: | :----: | :----: | :----: |
| CRPS-Sum | 0.3622 | 0.3285 | 0.339 |  