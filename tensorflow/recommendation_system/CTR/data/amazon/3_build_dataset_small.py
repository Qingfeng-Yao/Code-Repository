# -*- coding:utf-8 -*-
import random
import pickle

random.seed(1234)

with open('remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
# 统计用户历史序列信息
n = 0
lens = 0
max_l = 0
min_l = 1000000
d = {0: 0, 1: 0, 2: 0} # 0: 序列长度不超过5；1: 序列长度不超过20；2: 序列长度超过20；该字典的key指明用户人群信息
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  n += 1 # 统计用户个数
  pos_list = hist['asin'].tolist()

  # 统计用户序列的最大值、最小值和平均值
  l = len(pos_list)-2
  if l<min_l:
    min_l = l
  if l>max_l:
    max_l = l
  lens += l
  # 统计用户人群信息
  if l<=5:
    d[0] += 1
    reviewerGroup = 0
  elif l>5 and l<=20:
    d[1] += 1
    reviewerGroup = 1
  else:
    d[2] += 1
    reviewerGroup = 2

  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  # 删除原始处理逻辑，训练集中的样本数与测试集相同，且每个样本添加用户人群信息
  k = len(pos_list)-2
  train_set.append((reviewerID, pos_list[:k], pos_list[k], 1, reviewerGroup)) 
  train_set.append((reviewerID, pos_list[:k], neg_list[k], 0, reviewerGroup))
  k = len(pos_list)-1
  test_set.append((reviewerID, pos_list[:k], pos_list[k], 1, reviewerGroup))
  test_set.append((reviewerID, pos_list[:k], neg_list[k], 0, reviewerGroup))

# 打印相关统计信息
print(max_l, min_l, n, lens/n) # 429 3 192403 6.779426516218562
print(d[0], d[1], d[2]) # 119022 66644 6737

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train{}, n_test{}".format(len(train_set), len(test_set)))
# 更改输出数据集的名字
with open('dataset_small_group.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



