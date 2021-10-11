### basic
- pytorch_cookbook: pytorch基础
- QA: 遇到的一些常见问题及解决方法
- 参考资料
    - [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
    - [lyhue1991/eat_pytorch_in_20_days](https://github.com/lyhue1991/eat_pytorch_in_20_days)
    - [Atcold/pytorch-Deep-Learning](https://github.com/Atcold/pytorch-Deep-Learning)
    - [pytorch/examples](https://github.com/pytorch/examples)
    - [rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models)
    - [ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

### pipeline
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

### 深度学习模型
- natural language processing