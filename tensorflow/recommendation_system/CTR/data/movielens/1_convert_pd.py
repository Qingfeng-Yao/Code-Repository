import pickle
import pandas as pd

def to_df(file_path):
      df = pd.read_csv(file_path)
      return df

ratings_df = to_df('ratings.csv')
# print(len(ratings_df))
ratings_df = ratings_df[ratings_df['rating']>3.0]
# print(len(ratings_df))
ratings_df = ratings_df.reset_index(drop=True)
with open('ratings.pkl', 'wb') as f:
    pickle.dump(ratings_df, f)
meta_df = to_df('movies.csv')
# print(len(meta_df))
meta_df = meta_df[meta_df['movieId'].isin(ratings_df['movieId'].unique())]
# print(len(meta_df))
meta_df = meta_df.reset_index(drop=True)
with open('meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f)
