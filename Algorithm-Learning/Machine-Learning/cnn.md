## CNN
- LeCun et al. NIPS 1989
- Inspired by (Hubel & Wiesel 1962) & (Fukushima 1982)
	- simple cells detect local features
	- complex cells “pool” the outputs of simple cells within a retinotopic neighborhood
- LeNet5, vintage 1990
	- Filters-tanh → pooling → filters-tanh → pooling → filters-tanh
- depth inflation
	- VGG | Simonyan 2013
	- GoogLeNet| Szegedy 2014
	- ResNet | He et al. 2015
	- DenseNet | Huang et al 2017
- 广泛应用于计算机视觉`CV`、语音`speech`和自然语言处理`NLP`
- 解决高维学习问题的有力架构
- 主要假设：data(images, videos, speech) is compositional(local、stationary/shared patterns、hierarchical/multi-scale)
	- CNN利用compositionality结构(抽取compositional特征)(多层架构==数据的compositional结构)
	- 数据域：data(speech/words/sentences, images)存在于1D、2D的欧式域(grids)中，这些域具有强的规则空间结构
		- 数据表示的坐标具有网格结构，并且这些坐标中的要研究的数据相对于网格具有平移不变性(一个区域中有用的局部特征可能在其他区域中也有用)(有效重用具有可学习参数的局部filters，将其应用于所有输入位置上)
		- 所有的CNN操作(如卷积、池化/子采样)都是`mathematically well defined`和`fast`
		- 图域：非欧式域，具有不规则空间结构。图可以编码复杂的几何结构，可以用谱图理论等强大的数学工具来研究
		- 上述欧式域中的数据和非欧式域中的数据统称为`结构化数据`(由定义在某种结构的域上的采样实值函数组成，如在图的节点上定义的标量函数)
- 三种机制
	- 局部接受场local receptive field
	- 权重共享
	- 子采样subsampling/池化pooling

## 卷积层(for grids)
- 定义卷积：
	- 定义一：`template matching`，其中template指的是参数矩阵
	- 定义二：卷积理论(两个函数的卷积的傅立叶变换=两个函数傅立叶变换的pointwise product，即两个函数的卷积=两个函数傅立叶变换的pointwise product的逆变换)
- 单元组成平面，每个平面称为特征地图
	- 特征地图中的每个单元仅从图像的一个小的子区域获取输入(利用图像的一个关键性质即邻近像素的相关性比较远像素强)，并且特征地图中的所有单元都被约束为共享相同的权重值(在图像的一个区域中有用的局部特征可能在图像的其他区域中有用)
	- 特征地图中的所有单元(特征检测器)都会在输入图像的不同位置检测到相同的模式
	- 卷积层中通常有多个特征地图，每个特征地图都有自己的权重参数
## 子采样层
- 每个子采样单元可从对应特征地图中的单元区域中获取输入，并计算这些输入的平均值/最大值，乘以自适应权重，再加上自适应偏差参数，然后使用sigmoidal非线性激活函数进行变换
	- 子采样层中的单元的响应对输入空间的相应区域中的图像的小位移相对不敏感
	- 空间分辨率resolution的逐渐降低随后由特征数量的增加来补偿

