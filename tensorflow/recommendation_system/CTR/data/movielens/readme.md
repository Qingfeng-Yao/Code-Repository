## movielens
### ���ݼ��
- ԭʼ�ļ�`ratings.csv`��`movies.csv`��������[��](https://grouplens.org/datasets/movielens/20m/)�����ݴ���ο�����Deep Interest Network for Click-Through Rate Prediction
- ratings.csv��ÿһ������1,2,3.5,1112486027��`����20000263��`�����е�һ��Ϊ`userId`���ڶ���Ϊ`movieId`��������Ϊ`rating`��������Ϊ`timestamp`
- movies.csv��ÿһ������1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy��`����27278��`�����е�һ��Ϊ`movieId`���ڶ���Ϊ`title`��������Ϊ`genres`

### ���ݴ���
- 1_convert_pd.py: ʹ��pd�ֱ��ȡratings.csv��movies.csv�����ݣ����ڵ�һ���ļ���rating���µ���ֵ����>=4��дΪ1��0�����ڵڶ����ļ�ֻѡȡ��һ���ļ��г��ֵ�movieId���õ���22884�У����ֱ�д��`ratings.pkl`��`meta.pkl`
- `2_remap_id.py`: ��`ratings.csv`��ÿһ����¼ֻ����`['userId', 'movieId', 'timestamp']`������`userId`, `movieId`��ӦֵתΪ��������`meta.pkl`��ÿһ����¼ֻ����`['movieId', 'genres']`������genresֻȡ���һ���࣬����`movieId`, `genres`��ӦֵתΪ������ͳ��`�û���138362����Ӱ��22884�������20��������12195566`�����õ�`remap.pkl`����4������ɣ��ֱ���ratings_df(���û�ID������ʱ���ź�������ּ�¼)��cate_list(�������ź���ĵ�Ӱ��Ӧ����ɵ��б�)������Ԫ��(һ����user_count, item_count, cate_count, example_count��ɣ���һ����item_key, cate_key, user_key�����б���ɣ������б�ֱ��ӦΨһ�ĵ�ӰԭʼID��Ψһ�����ԭʼID��Ψһ���û�ԭʼID)
- `3_build_dataset.py`: ���û����з��飬��ÿ���û����ɺ�������(������N��)һ����Ŀ�ĸ�����������ѵ������ʹ��ǰN-1�������������ڲ��Լ�ʹ������N������������ΪҪ�����г�������Ϊ3������ɸѡ�����û���Ϊ138078��ѵ������ÿһ������Ϊһ��Ԫ�飬�������û�ID����ʷ�������б�(����ΪN-2)��Ŀ����(��)������1(��0)���û��ķ�����Ϣ�����Լ���ÿһ������ҲΪһ��Ԫ�飬�������û�ID����ʷ�������б�(����ΪN-1)��Ŀ����(��)������1(��0)���û��ķ�����Ϣ��shuffle���ݼ�������ѵ�����Ͳ��Լ���������ͬ������276156�����ݣ�����`dataset.pkl`
- `4_dump_tfrecord.py`: ��ѵ�����Ͳ��Լ�����ת����tfrecords��ʽ������`movielens_train.tfrecords`��`movielens_valid.tfrecords`��tfrecords��ʽҪ��ָ��ÿһ�����ݵ�feature�����ڸ����ݼ�������`'user_id', 'hist_item_list', 'hist_cate_list', 'hist_length', 'item', 'item_cate', 'target', 'user_group'`