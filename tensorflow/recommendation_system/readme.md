## �ο�����
- [DSXiangLi/CTR](https://github.com/DSXiangLi/CTR)

## ��������
- `python3.6`; `requirements.txt`
- `conda install cudatoolkit=10.0`
- `conda install cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/`

## ����
### CTR
- amazon(eletronics)
    - reviews_Electronics_5.json
        - ÿһ������`{'reviewerID': 'A2JXAZZI9PHK9Z', 'asin': '0594451647', 'reviewerName': 'Billy G. Noland "Bill Noland"', 'helpful': [3, 3], 'reviewText': 'I am using this with a Nook HD+. It works as described. The HD picture on my Samsung 52&#34; TV is excellent.', 'overall': 5.0, 'summary': 'HDMI Nook adapter cable', 'unixReviewTime': 1388707200, 'reviewTime': '01 3, 2014'}`����1689188�У��õ�reviews.pkl
        - ÿһ��ֻ����`['reviewerID', 'asin', 'unixReviewTime']`������'reviewerID', 'asin'��Ӧ��ֵתΪ����
    - meta_Electronics.json
        - ÿһ������`{'asin': '0558835155', 'description': 'Use these high quality headphones for internet chatting and enjoy the comfort and ease of the headphones with the microphone and in-line volume control.Works with: Skype msn AIM YAHOO! Windows Live', 'title': 'Polaroid Pbm2200 PC / Gaming Stereo Headphones With Microphone &amp; In-line Volume', 'price': 13.95, 'imUrl': 'http://ecx.images-amazon.com/images/I/21rEirndRLL.jpg', 'categories': [['Electronics', 'Accessories & Supplies', 'Audio & Video Accessories', 'Headphones']]}`����498196�У���asinȥ�صõ�meta.pkl����63001��
        - ÿһ��ֻ����`['asin', 'categories']`������categoriesֻȡ���һ���࣬����'asin', 'categories'��Ӧ��ֵתΪ����
    - remap.pkl: ��4������ɣ��ֱ���reviews_df��cate_list(ÿ��item��Ӧ����ɵ��б�)������Ԫ��(һ����user_count, item_count, cate_count, example_count��ɣ���һ����asin_key, cate_key, revi_key�����ֵ���ɣ������ֵ�ֱ��Ӧitemid����id���û�id)
        - �û���192403��item��63001������801��������1689188
    - ���ɺ�������һ����Ŀ�ĸ������������û����з��飻ÿ��ѵ�����Ͳ��Լ��е�ÿһ��������һ��Ԫ�飬�����û�id����ʷ���С�itemid�Լ��Ƿ�����(0��1)��dataset.pkl��train_set��test_set��ɣ�ǰ�߹���2608764�����߹���384806
    - �������amazon_train.tfrecords��amazon_valid.tfrecords
- DIN
    - ʹ��tf.estimator.Estimator������model_fnָ��ģ��(tf.estimator.EstimatorSpec)������configָ������(tf.estimator.RunConfig)
    - ģ�ͺ���������Ϊ: features, labels, mode, params���ֱ���������������ǩ��ģʽ��ģ�Ͳ���������ģʽ��tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT������
    - ģ�ͺ��������Ϊtf.estimator.EstimatorSpec
    - ģ��ѵ��: �ֱ���tf.estimator.TrainSpec��tf.estimator.EvalSpec��Ȼ���ģ��һ���뵽tf.estimator.train_and_evaluate��
    - ģ��Ԥ��: ����estimator.predict����