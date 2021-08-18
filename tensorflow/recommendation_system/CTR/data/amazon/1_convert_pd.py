import pickle
import pandas as pd

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      # if i == 5:
      #   print(line)
      #   print(df[i])
      i += 1
    print("{} {}".format(file_path, i))
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

reviews_df = to_df('reviews_Electronics_5.json')
with open('reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f)#, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('meta_Electronics.json')
# print(meta_df)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
# print(meta_df)
meta_df = meta_df.reset_index(drop=True)
with open('meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f)#, pickle.HIGHEST_PROTOCOL)