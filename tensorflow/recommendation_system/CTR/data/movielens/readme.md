## movielens
### 数据简介
- 原始文件`ratings.csv`和`movies.csv`，来自于[此](https://grouplens.org/datasets/movielens/20m/)，数据处理参考论文Deep Interest Network for Click-Through Rate Prediction
- ratings.csv中每一行形如1,2,3.5,1112486027，`共有20000263行`，其中第一栏为`userId`，第二栏为`movieId`，第三栏为`rating`，第四栏为`timestamp`
- movies.csv中每一行形如1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy，`共有27278行`，其中第一栏为`movieId`，第二栏为`title`，第三栏为`genres`

### 数据处理
- 1_convert_pd.py: 使用pd分别读取ratings.csv和movies.csv的内容，对于第一个文件将rating栏下的数值依据>=4重写为1或0；最后分别写入`ratings.pkl`和`meta.pkl`
- 2_remap_id.py: 对ratings.csv中每一条记录只保留`['userId', 'movieId', 'timestamp', 'rating']`，并将`userId`, `movieId`对应值转为索引；对`meta.pkl`中每一条记录只保留`['movieId', 'genres']`，其中genres只取最后一个类，并将`movieId`, `genres`对应值转为索引；统计`用户数138493，电影数27278，类别数20，样本数20000263`；最后得到`remap.pkl`，由4部分组成，分别是ratings_df(按用户ID和评分时间排好序的评分记录)、cate_list(按索引排好序的电影对应类组成的列表)和两个元组(一个由user_count, item_count, cate_count, example_count组成，另一个由item_key, cate_key, user_key三个列表组成，三个列表分别对应唯一的电影原始ID、唯一的类别原始ID和唯一的用户原始ID)
- 3_build_dataset.py: 按用户划分训练集和测试集，随机选择`100000个用户`为训练集(`共有100000个样本`)，剩下的`38493个用户`为测试集(`共有38493个样本`)，生成`dataset.pkl`
- 4_dump_tfrecord.py: 将训练集和测试集数据转换成tfrecords格式，生成`movielens_train.tfrecords`和`movielens_valid.tfrecords`；tfrecords格式要求指明每一条数据的feature，对于该数据集，包括`'user_id', 'hist_item_list', 'hist_cate_list', 'hist_length', 'item', 'item_cate', 'target'`