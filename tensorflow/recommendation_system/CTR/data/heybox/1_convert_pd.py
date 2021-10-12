import pandas as pd
import pickle

data_time = '04-15'
full_path = '2021-'+data_time+'-contact.csv'

column_names = ['id','event_type','timestamp','os_type','event_id','device_id','topic_id','src','recommend_idx','link_tag','is_push','view_time','userid','al','page_tab','version','src_detailed']
data = pd.read_csv(full_path, header=None, names=column_names)
pos_data = data[['event_type', 'timestamp', 'event_id', 'topic_id', 'view_time', 'userid']]
neg_data = data[['event_type', 'timestamp', 'event_id', 'topic_id', 'userid']]

neg_data = neg_data.dropna()
neg_data = neg_data[neg_data['event_type']==14]
neg_data['event_type'] = neg_data['event_type'].map(lambda x: 0)
print(neg_data.shape)

pos_data = pos_data.dropna()
# pos samples: event_type=12 and view_time>5s
pos_data = pos_data[(data['event_type']==12)&(pos_data['view_time']>5)]
pos_data = pos_data[['event_type', 'timestamp', 'event_id', 'topic_id', 'userid']]
pos_data['event_type'] = pos_data['event_type'].map(lambda x: 1)
print(pos_data.shape)

final_data = pd.concat([pos_data, neg_data], axis=0, ignore_index=True)
print(final_data.shape)
with open('user_log.pkl', 'wb') as f:
	pickle.dump(final_data, f)