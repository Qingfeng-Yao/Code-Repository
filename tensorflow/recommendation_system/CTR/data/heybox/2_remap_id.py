import random
import pickle
import numpy as np

random.seed(1234)

with open('user_log.pkl', 'rb') as f:
  view_df = pickle.load(f)
  view_df['userid'] = view_df['userid'].map(lambda x: int(x))
  view_df['topic_id'] = view_df['topic_id'].map(lambda x: int(x))
  view_df['event_id'] = view_df['event_id'].map(lambda x: int(x))


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

user_map, user_key = build_map(view_df, 'userid')
topic_map, topic_key = build_map(view_df, 'topic_id')
post_map, post_key = build_map(view_df, 'event_id')

user_count, post_count, topic_count, example_count =\
    len(user_map), len(post_map), len(topic_map), view_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, post_count, topic_count, example_count))

view_df = view_df.sort_values(['userid', 'timestamp'])
view_df = view_df.reset_index(drop=True)

with open('remap.pkl', 'wb') as f:
  pickle.dump(view_df, f)
  pickle.dump((user_count, post_count, topic_count, example_count), f)
  pickle.dump((user_key, topic_key, post_key), f)

