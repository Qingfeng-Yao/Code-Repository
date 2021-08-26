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
# ͳ���û���ʷ������Ϣ
n = 0
lens = 0
max_l = 0
min_l = 1000000
d = {0: 0, 1: 0, 2: 0} # 0: ���г��Ȳ�����50��1: ���г��Ȳ�����200��2: ���г��ȳ���200�����ֵ��keyָ���û���Ⱥ��Ϣ
for userId, hist in ratings_df.groupby('userId'):
  pos_list = hist['movieId'].tolist()

  # ͳ���û����е����ֵ����Сֵ��ƽ��ֵ
  l = len(pos_list)-2
  if l<1:
      continue
  n += 1 # ͳ����Ч�û�����
  if l<min_l:
    min_l = l
  if l>max_l:
    max_l = l
  lens += l
  # ͳ���û���Ⱥ��Ϣ
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

  # ѵ�����е�����������Լ���ͬ����ÿ����������û���Ⱥ��Ϣ
  k = len(pos_list)-2
  train_set.append((userId, pos_list[:k], pos_list[k], 1, userGroup)) 
  train_set.append((userId, pos_list[:k], neg_list[k], 0, userGroup))
  k = len(pos_list)-1
  test_set.append((userId, pos_list[:k], pos_list[k], 1, userGroup))
  test_set.append((userId, pos_list[:k], neg_list[k], 0, userGroup))

# ��ӡ���ͳ����Ϣ
print(max_l, min_l, n, lens/n) # 4166 1 138078 86.3204565535422
print(d[0], d[1], d[2]) # 78730 44712 14636

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train{}, n_test{}".format(len(train_set), len(test_set)))

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



