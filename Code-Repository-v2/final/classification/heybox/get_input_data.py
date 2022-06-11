# -*- coding:utf-8 -*-
import os
import json

raw_path = "raw_data"
input_path = "input_data"

if not os.path.exists(input_path):
    os.makedirs(input_path)

game_dict= {"彩虹六号围攻":"rainbow", "刀塔2":"daota2", "刀塔霸业":"daotabaye", "刀塔自走棋":"daotazizouqi", "怪物猎人世界":"monster", "盒友杂谈":"zatan", "绝地求生":"qiusheng", "炉石传说：魔兽英雄传":"lushi", "命运2":"mingyun2", "魔兽世界":"world", "手机游戏":"mobile", "守望先锋":"xianfeng", "数码硬件":"hardware", "英雄联盟":"union", "云顶之弈":"cloud", "主机游戏":"zhuji", "CS:GO":"csgo", "PC游戏":"pc"}

input_dict = {}
for root, dirs, files in os.walk(raw_path):
    if len(dirs) != 0:
        print(len(dirs))
    for f in files:
        if f[-4:] == 'json':
            file_path = root +'/'+ f
            assert f.split('.')[0] not in input_dict # 确保每个文档只属于一类，且每类下没有重复
            input_dict[f.split('.')[0]] = {}

            cate = game_dict[root.split('/')[-1]]
            input_dict[f.split('.')[0]]["label"] = cate

            with open(file_path,'r',encoding='utf-8') as load_f:
                text = load_f.read()
                input_dict[f.split('.')[0]]["content"] = text

            
print("total documents: {}".format(len(input_dict)))
with open(input_path+'/data.json','w',encoding='utf-8') as w_f:
    json.dump(input_dict, w_f, indent=4)
                                
