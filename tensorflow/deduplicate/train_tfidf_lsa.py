import json
import TfidfLsa
import os
import numpy as np

from util import compute_metrics, cmp_gt

data_path = "data/heybox/input_data/data.json"
threshold = 0.6 # 0.6 | 0.8
is_cmp = True
lsa_n_components = 200
gt_path = "data/heybox/ground_truth/"
out_path = "data/heybox/cmp/tfidf_lsa_{}/".format(threshold)

np.random.seed(1)

with open(data_path, "r") as f:
    query_data = json.load(f)

deduplication = TfidfLsa.Deduplication(threshold=threshold)
bag, query_items = deduplication.deduplicate(query_data, lsa_n_components)

# test on ground truth
gt_dirs = [d for d in os.listdir(gt_path) if '.txt' in d]
print("gt classes: {}".format(len(gt_dirs)))
print("pred classes: {}".format(len(bag)))
ave_p,ave_r,f1 = compute_metrics(gt_dirs, gt_path, bag, query_items)
print("recall: %f, precision: %f, f1-score: %f" % (ave_r,ave_p,f1))

# output different pred results from ground truth
if is_cmp:
    cmp_gt(bag, gt_dirs, out_path, gt_path, query_items)