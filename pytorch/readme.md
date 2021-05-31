## 内容
- pytorch_cookbook: pytorch基础
- basic_models
    - 线性回归
    - 逻辑回归
    - 前馈神经网络
    - 卷积神经网络
    - 残差网络
    - 循环神经网络: 
        - LSTM
        - 双向
    - 语言模型
        - LSTM
    - 生成对抗模型

## Pipeline
- 定义参数
    - 设置设备(模型和数据)
    - 输入输出大小、隐含大小、批大小
    - 迭代次数、学习率
- 数据下载以及data_loader定义
    - 自定义数据、标准数据
- 模型定义
- 损失函数和优化器定义
    - 损失函数可选MSE、CrossEntropy
    - 优化器可选SGD、Adam
        - 其中学习率可更新降低，即若干个epoch更新一次
- 模型训练
    - 进行迭代(epoch和batch)
    - 前向传播和损失计算
    - 反向传播和优化: optimizer.zero_grad() | loss.backward() | optimizer.step()
- 模型测试
    - 不计算梯度: model.eval()/with torch.no_grad()
    - batch迭代和前向传播
    - 指标计算: accuracy
- 保存模型

## 参考资料
- [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
- [Atcold/pytorch-Deep-Learning](https://github.com/Atcold/pytorch-Deep-Learning)