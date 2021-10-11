## movielens
### ���ݼ��
- ԭʼ�ļ�`ratings.csv`��`movies.csv`��������[��](https://grouplens.org/datasets/movielens/20m/)�����ݴ���ο�����Deep Interest Network for Click-Through Rate Prediction
- ratings.csv��ÿһ������1,2,3.5,1112486027��`����20000263��`�����е�һ��Ϊ`userId`���ڶ���Ϊ`movieId`��������Ϊ`rating`��������Ϊ`timestamp`
- movies.csv��ÿһ������1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy��`����27278��`�����е�һ��Ϊ`movieId`���ڶ���Ϊ`title`��������Ϊ`genres`

### ���ݴ���
- 1_convert_pd.py: ʹ��pd�ֱ��ȡratings.csv��movies.csv�����ݣ����ڵ�һ���ļ���rating���µ���ֵ����>=4��дΪ1��0�����ֱ�д��`ratings.pkl`��`meta.pkl`
- 2_remap_id.py: ��ratings.csv��ÿһ����¼ֻ����`['userId', 'movieId', 'timestamp', 'rating']`������`userId`, `movieId`��ӦֵתΪ��������`meta.pkl`��ÿһ����¼ֻ����`['movieId', 'genres']`������genresֻȡ���һ���࣬����`movieId`, `genres`��ӦֵתΪ������ͳ��`�û���138493����Ӱ��27278�������20��������20000263`�����õ�`remap.pkl`����4������ɣ��ֱ���ratings_df(���û�ID������ʱ���ź�������ּ�¼)��cate_list(�������ź���ĵ�Ӱ��Ӧ����ɵ��б�)������Ԫ��(һ����user_count, item_count, cate_count, example_count��ɣ���һ����item_key, cate_key, user_key�����б���ɣ������б�ֱ��ӦΨһ�ĵ�ӰԭʼID��Ψһ�����ԭʼID��Ψһ���û�ԭʼID)
- 3_build_dataset.py: ���û�����ѵ�����Ͳ��Լ������ѡ��`100000���û�`Ϊѵ����(`����100000������`)��ʣ�µ�`38493���û�`Ϊ���Լ�(`����38493������`)������`dataset.pkl`
- 4_dump_tfrecord.py: ��ѵ�����Ͳ��Լ�����ת����tfrecords��ʽ������`movielens_train.tfrecords`��`movielens_valid.tfrecords`��tfrecords��ʽҪ��ָ��ÿһ�����ݵ�feature�����ڸ����ݼ�������`'user_id', 'hist_item_list', 'hist_cate_list', 'hist_length', 'item', 'item_cate', 'target'`