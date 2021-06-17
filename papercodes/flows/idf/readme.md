## 参考资料
- 2019 | NIPS | Integer Discrete Flows and Lossless Compression | Emiel Hoogeboom et al.
- [jornpeters/integer_discrete_flows](https://github.com/jornpeters/integer_discrete_flows)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
- 命令运行: 
	- CIFAR10: `python3 main_experiment.py --n_flows 8 --n_levels 3 --n_channels 512 --coupling_type densenet --densenet_depth 12 --n_mixtures 5 --splitprior_type densenet`
	- ImageNet32:
	- ImageNet64:
- 压缩性能: bpd

## 运行结果