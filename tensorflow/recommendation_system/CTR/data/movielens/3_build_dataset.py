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
# 统计用户历史序列信息
n = 0
lens = 0
max_l = 0
min_l = 1000000
d = {0: 0, 1: 0, 2: 0} # 0: 序列长度不超过50；1: 序列长度不超过200；2: 序列长度超过200；该字典的key指明用户人群信息
for userId, hist in ratings_df.groupby('userId'):
  pos_list = hist['movieId'].tolist()

  # 统计用户序列的最大值、最小值和平均值
  l = len(pos_list)-2
  if l<1:
      continue
  n += 1 # 统计有效用户个数
  if l<min_l:
    min_l = l
  if l>max_l:
    max_l = l
  lens += l
  # 统计用户人群信息
  if l<=50:
    d[0] += 1
    userGroup = 0
  elif l>50 and l<=200:
    d[1] += 1
    userGroup = 1
  else:
    d[2] += 1
    userGroup = 2

  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  # 训练集中的样本数与测试集相同，且每个样本添加用户人群信息
  k = len(pos_list)-2
  train_set.append((userId, pos_list[:k], pos_list[k], 1, userGroup)) 
  train_set.append((userId, pos_list[:k], neg_list[k], 0, userGroup))
  k = len(pos_list)-1
  test_set.append((userId, pos_list[:k], pos_list[k], 1, userGroup))
  test_set.append((userId, pos_list[:k], neg_list[k], 0, userGroup))

# 打印相关统计信息
print(max_l, min_l, n, lens/n) # 4166 1 138078 86.3204565535422
print(d[0], d[1], d[2]) # 78730 44712 14636

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train{}, n_test{}".format(len(train_set), len(test_set)))

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



