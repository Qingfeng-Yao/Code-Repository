### basic
- tf_cookbook: tensorflow����
- QA: ������һЩ�������⼰�������

### pipeline
- ʹ��tf.estimator.Estimator������model_fnָ��ģ��(tf.estimator.EstimatorSpec)������configָ������(tf.estimator.RunConfig)
- ģ�ͺ���������Ϊ: features, labels, mode, params���ֱ���������������ǩ��ģʽ��ģ�Ͳ���������ģʽ��tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT������
- ģ�ͺ��������Ϊtf.estimator.EstimatorSpec
- ģ��ѵ��: �ֱ���tf.estimator.TrainSpec��tf.estimator.EvalSpec��Ȼ���ģ��һ���뵽tf.estimator.train_and_evaluate��
- ģ��Ԥ��: ����estimator.predict����

### ���ѧϰģ��
- recommendation system