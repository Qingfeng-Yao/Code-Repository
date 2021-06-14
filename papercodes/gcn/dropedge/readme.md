## 参考资料
- 2020 | ICLR | DropEdge: Towards Deep Graph Convolutional Networks on Node Classification | Yu Rong et al.
- [DropEdge/DropEdge](https://github.com/DropEdge/DropEdge)

## 源码运行
- 环境配置: `python3.6`; `requirements.txt`
	- 其中requests和tensorboard的版本会有冲突，我将tensorboard从2.1.0改成2.0.1
- 命令运行: 
	- 半监督: `sh script/semi-supervised/cora_GCN.sh`
	- 全监督: `sh script/supervised/cora_GCN.sh`
	- 上述命令中数据集和基础模型分别可在`[cora, citeseer, pubmed]`和`[GCN, IncepGCN, JKNet]`中选择，每一组对应的数据集和基础模型都含有原始执行和添加dropedge两条命令；默认所有模型均为两层
- 数据集格式与原始`GCN`一致，数据集划分与`Planetoid`一致，半监督设置与`GCN`一致，全监督设置与`FastGCN`和`ASGCN`一致
- 指标: accuracy

## 运行结果
半监督+cora
|  | GCN | JKNet |
| :----: | :----: | :----: |
| Orignal-2-layers | 81.10 | - |
| DropEdge-2-layers | 82.80 | - |
| Orignal-4-layers | - | 80.20 |
| DropEdge-4-layers | - | 83.30 |

`更多运行结果可参考源github`