from simhash import Simhash
import json
from collections import defaultdict
import os

from TfidfLsa.preprocess import preprocess
from util import compute_metrics, cmp_gt

data_path = "data/heybox/input_data/data.json"
threshold = 16
is_cmp = False
gt_path = "data/heybox/ground_truth/"
out_path = "data/heybox/cmp/simhash_{}/".format(threshold)

with open(data_path, "r") as f:
    query_data = json.load(f)

q_data = preprocess.cutWord(query_data)
query_ids = []
query_texts = []
query_items = q_data.items()
for idx, (id_, v) in enumerate(query_items):
    query_ids.append(id_)
    query_texts.append(v["text"])

bag = defaultdict(list)
for query_idx, (q_id, v) in enumerate(query_items):
    if len(bag) == 0:
        bag[q_id].append(query_idx)
        continue
        
    score = 1e9
    sim_bid = ""
    for bid, group in bag.items():
        for idx in group:
            sim = Simhash(query_texts[query_idx]).distance(Simhash(query_texts[idx]))
            if sim<score:
                score = sim
                sim_bid = bid
    if score<threshold:
        bag[sim_bid].append(query_idx)
    else:
        bag[q_id].append(query_idx)

query_items = list(query_items)

# test on ground truth
gt_dirs = [d for d in os.listdir(gt_path) if '.txt' in d]
print("gt classes: {}".format(len(gt_dirs)))
print("pred classes: {}".format(len(bag)))
ave_p,ave_r,f1 = compute_metrics(gt_dirs, gt_path, bag, query_items)
print("recall: %f, precision: %f, f1-score: %f" % (ave_r,ave_p,f1))

# output different pred results from ground truth
if is_cmp:
    cmp_gt(bag, gt_dirs, out_path, gt_path, query_items)
