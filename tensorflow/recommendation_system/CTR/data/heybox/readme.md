## heybox
### 数据简介
- 原始文件`2021-04-15-contact.csv`为不同时间下的用户日志数据
- 2021-04-15-contact.csv中每一行形如59291627734, 14, 2021-04-15 15:30:16, 1, 57105714, 8CA23761-06FE-43E7-B850-CACFDBD457DC, NaN, 4, 11, 0, 0, NaN, 18615050, NaN, NaN, 1.3.156, news_feeds_PUBG，对应`'id','event_type','timestamp','os_type','event_id','device_id','topic_id','src','recommend_idx','link_tag','is_push','view_time','userid','al','page_tab','version','src_detailed'`，其中event_type为行为类型(12为帖子停留时间，14为社区列表出现)，event_id为帖子ID，topic_id为帖子对应的社区ID，view_time为停留时间。`共78129046行`

### 数据处理
- 1_convert_pd.py: 仅保留2021-04-15-contact.csv中的'event_type', 'timestamp', 'event_id', 'topic_id', 'view_time', 'userid'并去除空值，然后以`event_type=12且view_time>5s`获得正样本(`共4439353行`)；负样本则只保留2021-04-15-contact.csv中的'event_type', 'timestamp', 'event_id', 'topic_id', 'userid'并去除空值，以以`event_type=14`获得(`共50945691行`)；最后将正负样本连接起来得到`user_log.pkl`
- 2_remap_id.py: 将'event_id', 'topic_id', 'userid'都转成索引，统计`用户数637278，帖子数216329，社区数4291，总样本数55385044`；最后得到`remap.pkl`，由3部分组成，分别是view_df(按用户ID和时间排好序的记录)和两个元组(一个由user_count, item_count, cate_count, example_count组成，另一个由user_key, topic_key, post_key三个列表组成，三个列表分别对应唯一的用户原始ID、唯一的社区原始ID和唯一的帖子原始ID)
- 3_build_dataset.py: 按用户分组并划分训练集和测试集，随机选择`400000个用户`为训练集(`共有265827个样本`)，剩下的`237278个用户`为测试集(`共有172984个样本`)，生成`dataset.pkl`
- 4_dump_tfrecord.py: 将训练集和测试集数据转换成tfrecords格式，生成`heybox_train.tfrecords`和`heybox_valid.tfrecords`；tfrecords格式要求指明每一条数据的feature，对于该数据集，包括`'user_id', 'hist_item_list', 'hist_cate_list', 'hist_length', 'item', 'item_cate', 'target'`