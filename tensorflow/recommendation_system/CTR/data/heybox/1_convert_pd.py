import pandas as pd
import pickle

data_time = '04-15'
full_path = '2021-'+data_time+'-contact.csv'

column_names = ['id','event_type','timestamp','os_type','event_id','device_id','topic_id','src','recommend_idx','link_tag','is_push','view_time','userid','al','page_tab','version','src_detailed']
data = pd.read_csv(full_path, header=None, names=column_names)
data = data[['event_type', 'timestamp', 'event_id', 'topic_id', 'view_time', 'userid']]
data = data.dropna()

# pos samples: event_type=12 and view_time>5s
data = data[(data['event_type']==12)&(data['view_time']>5)]
data = data.reset_index(drop=True)
with open('user_log.pkl', 'wb') as f:
	pickle.dump(data, f)