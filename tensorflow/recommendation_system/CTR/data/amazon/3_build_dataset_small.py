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
# ͳ���û���ʷ������Ϣ
n = 0
lens = 0
max_l = 0
min_l = 1000000
d = {0: 0, 1: 0, 2: 0} # 0: ���г��Ȳ�����5��1: ���г��Ȳ�����20��2: ���г��ȳ���20�����ֵ��keyָ���û���Ⱥ��Ϣ
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  n += 1 # ͳ���û�����
  pos_list = hist['asin'].tolist()

  # ͳ���û����е����ֵ����Сֵ��ƽ��ֵ
  l = len(pos_list)-2
  if l<min_l:
    min_l = l
  if l>max_l:
    max_l = l
  lens += l
  # ͳ���û���Ⱥ��Ϣ
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

  # ɾ��ԭʼ�����߼���ѵ�����е�����������Լ���ͬ����ÿ����������û���Ⱥ��Ϣ
  k = len(pos_list)-2
  train_set.append((reviewerID, pos_list[:k], pos_list[k], 1, reviewerGroup)) 
  train_set.append((reviewerID, pos_list[:k], neg_list[k], 0, reviewerGroup))
  k = len(pos_list)-1
  test_set.append((reviewerID, pos_list[:k], pos_list[k], 1, reviewerGroup))
  test_set.append((reviewerID, pos_list[:k], neg_list[k], 0, reviewerGroup))

# ��ӡ���ͳ����Ϣ
print(max_l, min_l, n, lens/n) # 429 3 192403 6.779426516218562
print(d[0], d[1], d[2]) # 119022 66644 6737

random.shuffle(train_set)
random.shuffle(test_set)
print("n_train{}, n_test{}".format(len(train_set), len(test_set)))
# ����������ݼ�������
with open('dataset_small_group.pkl', 'wb') as f:
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)



