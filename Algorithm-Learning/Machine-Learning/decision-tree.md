## 决策树
- 用于估计离散值目标函数；学习到的函数被表示为一棵决策树
- if-then Rules
	- 需要确定决策需要的属性

### ID3
- Entropy: 在一个封闭系统中，无序或随机`disorder or randomness`的测度
	- 在一个同质`homogenous`系统中，Entropy为0；若两个类有相同的实例数，则Entropy为1
- Conditional Entropy：给定属性X，系统的Entropy
- 信息增益`Information Gain`：分裂之前系统的熵-分裂之后系统的熵
	- 选择信息增益大的属性
	
### C4.5
- 信息增益比`Information-Gain Ratio`

### continuous-valued attributes
- 选定阈值划分区间，将连续属性转成布尔属性

### 未知属性值

### 分类回归树(CART)
- 分类树
- 回归树

### pruning

### 随机森林
- 使用训练数据集建立多个决策树从而组成森林
- 新的实例输入时，使森林中的每个决策树都进行投票；对于新实例的最终分类输出结果为森林中所有决策树输出最多的类别
