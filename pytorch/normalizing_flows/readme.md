## 参考资料
- [ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows)

## 内容
- 实现的模型
	- MAF
	- Glow
	- Real NVP
- 使用的数据集来自原始MAF库，包括`POWER`, `GAS`, `HEPMASS`, `MINIBONE`和`BSDS300`
- 命令执行: `python3 main.py --dataset POWER`
	- 环境配置: `python3.6`; `requirements.txt`
	- 需要事先将数据下载到`data`下 | [下载链接](https://zenodo.org/record/1161203#.YMdLiGQzb9G)
- 指标: 对数似然(in nats)

## 运行结果
|  | POWER | GAS | HEPMASS | MINIBOONE | BSDS300 | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| MAF(5) |