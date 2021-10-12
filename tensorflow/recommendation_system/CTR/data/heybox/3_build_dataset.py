# -*- coding:utf-8 -*-
import random
import pickle

random.seed(1234)

with open('remap.pkl', 'rb') as f:
  view_df = pickle.load(f)
  user_count, post_count, topic_count, example_count = pickle.load(f)

train_set = []
test_set = []
n_train_user = 0
n_test_user = 0

for reviewerID, hist in view_df.groupby('userid'):
  his_list = hist['event_id'].tolist()
  topic_list = hist['topic_id'].tolist() 
  label_list = hist['event_type'].tolist()

  is_train = random.randint(0, 1)
  if is_train and n_train_user < 400000:
    n_train_user += 1
    pos_list = []
    pos_topic_list = []
    for i in range(len(his_list[:-1])):
      if label_list[i] == 1:
        pos_list.append(his_list[i])
        pos_topic_list.append(topic_list[i])
    if len(pos_list)>0:
      train_set.append((reviewerID, pos_list, pos_topic_list, his_list[-1], topic_list[-1], label_list[-1])) 
  else:
    if n_test_user < 237278:
      n_test_user += 1
      pos_list = []
      pos_topic_list = []
      for i in range(len(his_list[:-1])):
        if label_list[i] == 1:
          pos_list.append(his_list[i])
          pos_topic_list.append(topic_list[i])
      if len(pos_list)>0:
       test_set.append((reviewerID, pos_list, pos_topic_list, his_list[-1], topic_list[-1], label_list[-1]))  
    else:
      n_train_user += 1
      pos_list = []
      pos_topic_list = []
      for i in range(len(his_list[:-1])):
        if label_list[i] == 1:
          pos_list.append(his_list[i])
          pos_topic_list.append(topic_list[i])
      if len(pos_list)>0:
        train_set.append((reviewerID, pos_list, pos_topic_list, his_list[-1], topic_list[-1], label_list[-1])) 

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train_user{}, n_test_user{}".format(n_train_user, n_test_user))
print("n_train{}, n_test{}".format(len(train_set), len(test_set))) 

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



