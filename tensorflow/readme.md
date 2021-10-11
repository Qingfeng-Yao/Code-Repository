### basic
- tf_cookbook: tensorflow基础
- QA: 遇到的一些常见问题及解决方法

### pipeline
- 使用tf.estimator.Estimator，参数model_fn指明模型(tf.estimator.EstimatorSpec)，参数config指明配置(tf.estimator.RunConfig)
- 模型函数的输入为: features, labels, mode, params，分别是批特征、批标签、模式和模型参数，其中模式由tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT来定义
- 模型函数的输出为tf.estimator.EstimatorSpec
- 模型训练: 分别定义tf.estimator.TrainSpec和tf.estimator.EvalSpec，然后和模型一起传入到tf.estimator.train_and_evaluate中
- 模型预测: 调用estimator.predict方法

### 深度学习模型
- recommendation system