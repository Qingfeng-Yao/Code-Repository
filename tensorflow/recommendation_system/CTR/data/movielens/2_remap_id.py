import random
import pickle
import numpy as np

random.seed(1234)

with open('ratings.pkl', 'rb') as f:
  ratings_df = pickle.load(f)
  ratings_df = ratings_df[['userId', 'movieId', 'timestamp', 'rating']]

with open('meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  meta_df = meta_df[['movieId', 'genres']]
  meta_df['genres'] = meta_df['genres'].map(lambda x: x.split('|')[-1])


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

item_map, item_key = build_map(meta_df, 'movieId')
cate_map, cate_key = build_map(meta_df, 'genres')
user_map, user_key = build_map(ratings_df, 'userId')

user_count, item_count, cate_count, example_count =\
    len(user_map), len(item_map), len(cate_map), ratings_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('movieId')
meta_df = meta_df.reset_index(drop=True)
ratings_df['movieId'] = ratings_df['movieId'].map(lambda x: item_map[x])
ratings_df = ratings_df.sort_values(['userId', 'timestamp'])
ratings_df = ratings_df.reset_index(drop=True)

cate_list = [meta_df['genres'][i] for i in range(len(item_map))]
cate_list = np.array(cate_list, dtype=np.int32)

with open('remap.pkl', 'wb') as f:
  pickle.dump(ratings_df, f)
  pickle.dump(cate_list, f)
  pickle.dump((user_count, item_count, cate_count, example_count),
              f)
  pickle.dump((item_key, cate_key, user_key), f)

