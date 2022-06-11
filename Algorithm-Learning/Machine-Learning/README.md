## 模式识别
- 通过使用计算机算法自动发现(机器学习)数据中的规律`regularity`，然后使用这些规律采取行动，如分类
- 使用训练集(如数据特征和标签)来调模型(机器学习算法)的参数，或者说在训练(学习)期间确定函数的确定形式
  - 原始输入变量一般会经过预处理，转变成某个新空间中的变量，使问题简化(也可加速计算)；`预处理阶段`也称为`特征抽取`，可被看成`降维`；其中特征抽取一般是手动的
  - 通过拟合训练数据来确定参数值：最小化误差函数
  - 模型对比/模型选择：如选择多项式的阶M，M过小不能很好表示原函数；M过大虽然误差为零，但拟合的曲线振荡过大(此时系数绝对值过大，相应的多项式函数精确地匹配每个数据点，但在数据点之间/特别是在范围的末端附近，函数显示出大振荡。表明M值越大的更灵活的多项式越来越适应目标值上的随机噪声)，也不能很好表示原函数，此时`过拟合`
    - 当数据集变大，可减少过拟合问题
    - 参数数目不一定是模型复杂度的最合适的测度
    - 数据集大小有限，同时希望使用相对复杂和灵活模型，可使用正则化`regularization`来控制过拟合现象(加一个惩罚项到误差函数，为了阻止系数达到大值)。最简单的惩罚项是所有系数的平方和。统计学上称为shrinkage方法(减少系数值)，使用二次正则项称为ridge regression，在神经网络中称为weight decay。但正则化参数过大，拟合也会变差。该参数控制模型的有效复杂度，从而确定过拟合的程度
  - 确定模型复杂度的合适值：将可用数据划分成一个训练集和一个验证集(也叫hold-out set)，后者可用来优化模型复杂度(M或正则化参数)，但这浪费宝贵的训练数据
- 泛化：能够对新样例作出准确预测的能力
  - 难点：数据集有限；数据有噪声 --> 使得给定一个新输入，适当的目标值存在不确定性 --> 概率论提供一个框架，以精确和定量的方式表达这种不确定性；决策论允许我们利用这种概率表示，以便根据适当的标准做出最佳的预测
- 有监督学习：训练集中的每一个样例由输入向量和目标向量组成；输出离散是`分类`，输出连续是`回归`
- 无监督学习：训练集中的每一个样例只由输入向量组成；目标可以是发现数据中相似样例的groups(clustering)、确定数据在输入空间中的分布(density estimation)、将数据从高维空间映射到二、三维空间(visualization)
- 强化学习：在给定的situation下，找到要采取的合适的actions，使得reward最大；与有监督学习相比，这里的学习算法没有给出最优输出的样例，而是必须通过反复试验`trial and error`来发现它们
  - 有一系列的states和actions，在这些状态和动作中，学习算法与其环境交互；当前动作不仅影响立即的奖励，还影响后续所有步骤的奖励
  - 强化学习的一般特征是在`exploration`(系统尝试新类型的actions来看看它们是否有效)和`exploitation`(系统利用已知能产生高reward的actions)两者中进行`trade-off`
  
### 误差函数
- 平方和误差函数
  - 寻找模型参数的最小二乘法(least squares approach)代表了最大似然的一个具体情况，并且过拟合问题可以理解为最大似然的一个一般性质
  - 采用贝叶斯方法可以避免过拟合问题。从贝叶斯的角度来看，使用参数数量大大超过数据点数量的模型并不困难。事实上，在贝叶斯模型中，有效参数的数量自动适应数据集的大小

### 线性模型
- 未知参数的线性函数称为线性模型，如多项式

## 深度学习
- Inspiration for Deep Learning: The Brain!
- 与传统机器学习需要手动进行特征抽取相比，深度学习使得特征也是可学习的(包括`low-level`、`mid-level`和`high-level`的特征，引出多层神经网络)
  - Multiple Layers of simple units
  - Each units computes a weighted sum of its inputs
  - Weighted sum is passed through a non-linear function
  - The learning algorithm changes the weights(SGD；使用反向传播计算梯度)
- Deep Learning = Learning Representations/Features(End-to-end learning)
  - 一般的特征抽取：扩展表示`representation`的维数，使事物更可能成为线性可分`linearly separable`的
  - hierarchical representation：with increasing level of abstraction；Each stage is a kind of trainable feature transform
- Learning Representations of Data: Discovering & disentangling the independent explanatory factors
  - The Manifold Hypothesis:
    - Natural data lives in a low-dimensional (non-linear) manifold
    - Because variables in natural data are mutually dependent
  - Invariant Feature Learning
    - Embed the input non-linearly into a high(er) dimensional space；In the new space, things that were non separable may become separable；此时的特征是high-dim、unstable/non-smooth
    - Pool/Aggregate regions of the new space together；Bringing together things that are semantically similar；此时的特征是stable/invariant
- Deep machines对表示某些类的函数十分有效，尤其是那些涉及视觉识别`visual recognition`的
- Deep architectures有效的原因
  - 更多的层需要更多的序列计算更少的并行计算(更少的硬件)
  
### 传统神经网络
- 堆叠线性和非线性blocks
