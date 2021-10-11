# -*- coding:utf-8 -*-
import random
import pickle

random.seed(1234)

with open('remap.pkl', 'rb') as f:
  view_df = pickle.load(f)
  user_count, post_count, topic_count, example_count = pickle.load(f)
  topic_map = pickle.load(f)

train_set = []
test_set = []
# user hist info
n = 0
lens = 0
max_l = 0
min_l = 1000000

# length_max = 0 297
# length_min = 100000000 1
# for reviewerID, hist in view_df.groupby('userid'):
#   n += 1 # record user num
#   pos_list = hist['event_id'].tolist()
#   if len(pos_list) > length_max:
#     length_max = len(pos_list)
#   if len(pos_list) < length_min:
#     length_min = len(pos_list)
# print(n, length_max, length_min)

# record user seq length(max, min and average)
for reviewerID, hist in view_df.groupby('userid'):
  pos_list = hist['event_id'].tolist()
  topic_list = hist['topic_id'].tolist() 

  if len(pos_list) < 3:
    continue

  n += 1 # record real user num
  
  l = len(pos_list)
  if l<min_l:
    min_l = l
  if l>max_l:
    max_l = l
  lens += l

  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, post_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  # 删除原始处理逻辑(为了得到小的数据集)，训练集中的样本数与测试集相同
  k = len(pos_list)-2
  train_set.append((reviewerID, pos_list[:k], topic_list[:k], pos_list[k], topic_list[k], 1)) 
  train_set.append((reviewerID, pos_list[:k], topic_list[:k], neg_list[k], topic_map[neg_list[k]], 0))
  k = len(pos_list)-1
  test_set.append((reviewerID, pos_list[:k], topic_list[:k], pos_list[k], topic_list[k], 1))
  test_set.append((reviewerID, pos_list[:k], topic_list[:k], neg_list[k], topic_map[neg_list[k]], 0))

# 打印相关统计信息
print(n, max_l, min_l, lens/n) # 304997 297 3 13.652488385131656

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train{}, n_test{}".format(len(train_set), len(test_set))) # n_train609994, n_test609994

with open('dataset_small.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



