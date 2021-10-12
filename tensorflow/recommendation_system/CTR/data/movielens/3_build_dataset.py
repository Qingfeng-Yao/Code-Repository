# -*- coding:utf-8 -*-
import random
import pickle

random.seed(1234)

with open('remap.pkl', 'rb') as f:
  ratings_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
n_train_user = 0
n_test_user = 0

for userId, hist in ratings_df.groupby('userId'):
  his_list = hist['movieId'].tolist()
  label_list = hist['rating'].tolist()

  is_train = random.randint(0, 1)
  if is_train and n_train_user < 100000:
    n_train_user += 1
    pos_list = []
    for i in range(len(his_list[:-1])):
      if label_list[i] == 1:
        pos_list.append(his_list[i])
    if len(pos_list)>0:
      train_set.append((userId, pos_list, his_list[-1], label_list[-1])) 
  else:
    if n_test_user < 38493:
      n_test_user += 1
      pos_list = []
      for i in range(len(his_list[:-1])):
        if label_list[i] == 1:
          pos_list.append(his_list[i])
      if len(pos_list)>0:
        test_set.append((userId, pos_list, his_list[-1], label_list[-1]))  
    else:
      n_train_user += 1
      pos_list = []
      for i in range(len(his_list[:-1])):
        if label_list[i] == 1:
          pos_list.append(his_list[i])
      if len(pos_list)>0:
        train_set.append((userId, pos_list, his_list[-1], label_list[-1])) 
      
random.shuffle(train_set)
random.shuffle(test_set)
print("n_train_user{}, n_test_user{}".format(n_train_user, n_test_user))
print("n_train{}, n_test{}".format(len(train_set), len(test_set)))

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



