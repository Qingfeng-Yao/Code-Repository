## heybox
### ���ݼ��
- ԭʼ�ļ�`2021-04-15-contact.csv`Ϊ��ͬʱ���µ��û���־����
- 2021-04-15-contact.csv��ÿһ������59291627734, 14, 2021-04-15 15:30:16, 1, 57105714, 8CA23761-06FE-43E7-B850-CACFDBD457DC, NaN, 4, 11, 0, 0, NaN, 18615050, NaN, NaN, 1.3.156, news_feeds_PUBG����Ӧ`'id','event_type','timestamp','os_type','event_id','device_id','topic_id','src','recommend_idx','link_tag','is_push','view_time','userid','al','page_tab','version','src_detailed'`������event_typeΪ��Ϊ����(12Ϊ����ͣ��ʱ�䣬14Ϊ�����б����)��event_idΪ����ID��topic_idΪ���Ӷ�Ӧ������ID��view_timeΪͣ��ʱ�䡣`��78129046��`

### ���ݴ���
- 1_convert_pd.py: ������2021-04-15-contact.csv�е�'event_type', 'timestamp', 'event_id', 'topic_id', 'view_time', 'userid'��ȥ����ֵ��Ȼ����`event_type=12��view_time>5s`���������(`��4439353��`)����������ֻ����2021-04-15-contact.csv�е�'event_type', 'timestamp', 'event_id', 'topic_id', 'userid'��ȥ����ֵ������`event_type=14`���(`��50945691��`)����������������������õ�`user_log.pkl`
- 2_remap_id.py: ��'event_id', 'topic_id', 'userid'��ת��������ͳ��`�û���637278��������216329��������4291����������55385044`�����õ�`remap.pkl`����3������ɣ��ֱ���view_df(���û�ID��ʱ���ź���ļ�¼)������Ԫ��(һ����user_count, item_count, cate_count, example_count��ɣ���һ����user_key, topic_key, post_key�����б���ɣ������б�ֱ��ӦΨһ���û�ԭʼID��Ψһ������ԭʼID��Ψһ������ԭʼID)
- 3_build_dataset.py: ���û����鲢����ѵ�����Ͳ��Լ������ѡ��`400000���û�`Ϊѵ����(`����265827������`)��ʣ�µ�`237278���û�`Ϊ���Լ�(`����172984������`)������`dataset.pkl`
- 4_dump_tfrecord.py: ��ѵ�����Ͳ��Լ�����ת����tfrecords��ʽ������`heybox_train.tfrecords`��`heybox_valid.tfrecords`��tfrecords��ʽҪ��ָ��ÿһ�����ݵ�feature�����ڸ����ݼ�������`'user_id', 'hist_item_list', 'hist_cate_list', 'hist_length', 'item', 'item_cate', 'target'`