# -*- coding:utf-8 -*-
import os

import utils

data_dir = "raw_data"

def get_jsons(json_dir):
    json_list = []
    for j in os.listdir(json_dir):
        j_dir = os.path.join(json_dir,j)
        text = utils.readFile(j_dir)
        json_list.append(text)

    return json_list


rainbow_jsons = get_jsons(os.path.join(data_dir, "彩虹六号围攻"))
daota2_jsons = get_jsons(os.path.join(data_dir, "刀塔2"))
daotabaye_jsons = get_jsons(os.path.join(data_dir, "刀塔霸业"))
daotazizouqi_jsons = get_jsons(os.path.join(data_dir, "刀塔自走棋"))
monster_jsons = get_jsons(os.path.join(data_dir, "怪物猎人世界"))
zatan_jsons = get_jsons(os.path.join(data_dir, "盒友杂谈"))
qiusheng_jsons = get_jsons(os.path.join(data_dir, "绝地求生"))
lushi_jsons = get_jsons(os.path.join(data_dir, "炉石传说：魔兽英雄传"))
mingyun2_jsons = get_jsons(os.path.join(data_dir, "命运2"))
world_jsons = get_jsons(os.path.join(data_dir, "魔兽世界"))
mobile_jsons = get_jsons(os.path.join(data_dir, "手机游戏"))
xianfeng_jsons = get_jsons(os.path.join(data_dir, "守望先锋"))
hardware_jsons = get_jsons(os.path.join(data_dir, "数码硬件"))
union_jsons = get_jsons(os.path.join(data_dir, "英雄联盟"))
cloud_jsons = get_jsons(os.path.join(data_dir, "云顶之弈"))
zhuji_jsons = get_jsons(os.path.join(data_dir, "主机游戏"))
csgo_jsons = get_jsons(os.path.join(data_dir, "CS:GO"))
pc_jsons = get_jsons(os.path.join(data_dir, "PC游戏"))

print("Number of rainbow_jsons =", len(rainbow_jsons))
print("Number of daota2_jsons =", len(daota2_jsons))
print("Number of daotabaye_jsons =", len(daotabaye_jsons))
print("Number of daotazizouqi_jsons =", len(daotazizouqi_jsons))
print("Number of monster_jsons =", len(monster_jsons))
print("Number of zatan_jsons =", len(zatan_jsons))
print("Number of qiusheng_jsons =", len(qiusheng_jsons))
print("Number of lushi_jsons =", len(lushi_jsons))
print("Number of mingyun2_jsons =", len(mingyun2_jsons))
print("Number of world_jsons =", len(world_jsons))
print("Number of mobile_jsons =", len(mobile_jsons))
print("Number of xianfeng_jsons =", len(xianfeng_jsons))
print("Number of hardware_jsons =", len(hardware_jsons))
print("Number of union_jsons =", len(union_jsons))
print("Number of cloud_jsons =", len(cloud_jsons))
print("Number of zhuji_jsons =", len(zhuji_jsons))
print("Number of csgo_jsons =", len(csgo_jsons))
print("Number of pc_jsons =", len(pc_jsons))