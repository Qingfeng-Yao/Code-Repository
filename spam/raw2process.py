import os
import csv
import json
import re

import numpy as np 
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split

# 参数
data_str = "spam_2024" # ["spam_2024", "spam_http_1999", "spam_digit_1999"]
# ---------------

# def write_csv(text, label, path):
#     with open(path,'a+') as f:
#         csv_write = csv.writer(f)
#         data_row = [text,label]
#         csv_write.writerow(data_row)

if data_str == "spam_2024": # ["spam_2024", "spam_http_1999", "spam_digit_1999"]

    content = []
    meta_data = []
    doc_id = 0

    dict_text2label = {}
    csvFile = open("rawdata/{}.csv".format(data_str), "r")
    reader = csv.reader(csvFile)
    for item in reader:
        text = item[0].replace('\n', ' ')
        text = text.strip()
        if text == "":
            continue
        if item[1] not in dict_text2label:
            dict_text2label[item[1]] = [text]
        else:
            dict_text2label[item[1]].append(text)
    print("classes {}".format(len(dict_text2label)))

    # x_list = list(range(0, 1000, 50))
    # y_map = {}

    # csv_write_path = "rawdata/spam_digit.csv"

    # https = []
    # digits = []
    texts = []
    textid2len = {}
    for k, v in dict_text2label.items():
        print("{} : {}".format(k, len(v)))
        for t in v:
            if k == "普通内容":
                len_t = len(t)
                textid2len[len(texts)] = len_t
                texts.append(t)
    order_dict = sorted(textid2len.items(), key=lambda item:item[1], reverse=False)
    for (k, v) in order_dict:
        print("{} {}".format(v, texts[k]))






        # for t in v:
        #     matchs = re.findall(r"\d{7,}", t)

        #     if not matchs:
        #         len_t = len(t)
        #         c = Counter()
        #         for ch in t:
        #             c[ch] = c[ch] + 1
        #         pro = len(c)/len_t
        #         # if pro < 0.3 and len_t < 500:
        #             # print("{} {}: {} : {}".format(pro, len_t, k, t))
        #         if len_t > 1000:
        #             k_count = 0
        #             for kword in ["遇害", "出轨", "营业", "艹", "开房", "nt", "肉体", "作业", "泡我", "拿破仑", "爱迪生"]:
        #                 if kword in t:
        #                     k_count = 1
        #                     break
                    
        #             if k_count == 0:
        #                 print("{} : {} : {}".format(len_t, k, t))


            # len_t = len(t)
            # if len_t > 5000:
            #     print("{} : {} : {}".format(len_t, k, t))




        # lens_k = []
        # for t in v:
        #     lens_k.append(len(t))
        #     print("{} : {}".format(len(t), t))
        # print("平均长度及方差 {} : {} {}".format(k, np.mean(lens_k), np.var(lens_k)))
        # print("min {}  max {}".format(min(lens_k), max(lens_k)))
        # print("\n")


    #     k_map = {}
    #     for n in x_list:
    #         k_map[n] = 0
    #     for t in v:
    #         len_t = len(t)
    #         for i, n in enumerate(x_list):
    #             if len_t < n:
    #                 break
    #         k_map[x_list[i-1]] += 1
    #     order_k_map = sorted(k_map.items(), key=lambda item:item[0], reverse=False)
    #     y_list = [f for (c,f) in order_k_map]
    #     y_map[k] = y_list

    # # shapes = ['s-', 'o-'] #o-:圆形, s-:方形
    # colors = ['r', 'g']
    # i = 0
    # for k, v in dict_text2label.items():
    #     if k == "普通内容":
    #         plt.plot(x_list,y_map[k],color = colors[i],label="normal")
    #     else:
    #         plt.plot(x_list,y_map[k],color = colors[i],label="spam")
    #     i += 1
    
    # plt.xlabel("text length")#横坐标名字
    # plt.ylabel("doc number")#纵坐标名字
    # plt.legend(loc = "best")#图例
    # plt.show()

        # for t in v:
            # matchs_http = re.findall(r"http\S*", t)
            # matchs_www = re.findall(r"www\S*", t)
            # t1 = re.sub("\s", "", t)
            # matchs_digit = re.findall(r"\d{7,}", t1)
            # if matchs_http:
            #     # print("{} : {} : {}".format(k, t, matchs_http))
            #     # https.append((k,t))
            #     continue
            # if matchs_www:
            #     # print("{} : {} : {}".format(k, t, matchs_www))
            #     continue
            # if matchs_digit:
            #     # print("{} : {} : {}".format(k, t, matchs_digit))
            #     # digits.append((k,t))
            #     continue
    #         if k == "垃圾内容":
    #             len_t = len(t)
    #             textid2len[len(texts)] = len_t
    #             texts.append(t)


    # order_dict = sorted(textid2len.items(), key=lambda item:item[1], reverse=False)
    # for (k, v) in order_dict:
    #     print("{} {}".format(v, texts[k]))




            # print("{} : {}".format(k, t))

    # print(len(https), len(digits))
 


            