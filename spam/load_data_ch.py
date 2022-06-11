import csv
import os
import xlrd

from .utils import get_File_List, readFile, clean_text_ch, readStopwords
from . import setting

from sklearn.model_selection import train_test_split

# 中文预处理
# 使用jieba分词(自定义词典)
# 小写；分词后去停用词归一化
# 在分词之后将不符合"(?u)\\b\\w+\\b"这个模式的词进行归一化处理 一般这些词包括：表情、标点、图案、运算符、其他编码等符号(单个字符/非字母数字下划线) --> 归一化为"_symbol_"；百分数、小数等(部分非字母数字下划线) --> 归一化为"_number_"

# 包含垃圾和普通两类游戏领域文本
# 按8:2划分
class Spamgame_Dataset():
    def __init__(self, raw_data="datasets/data_dir/raw/spam_2024.csv", rm_seed=30):
        super().__init__()
        self.stopwords = readStopwords(setting.STOPWORDS_PATH)
        self.keywords = readStopwords(setting.KEYWORDS_PATH)
        self.numbers, self.symbols = set(), set()
        self.all_dict = {}

        # one_class = os.path.join(root, "垃圾内容")
        # another_class = os.path.join(root, "普通内容")
        # normFileList = get_File_List(another_class)
        # normFilelen=len(normFileList)
        # print("数据集中普通文本数量 {}".format(normFilelen))
        # spamFileList = get_File_List(one_class)
        # spamFilelen=len(spamFileList)
        # print("数据集中垃圾文本数量 {}".format(spamFilelen))
        extend_cur = os.path.splitext(raw_data)[-1][1:]

        normal_dict = {}
        normal_labels = []
        spam_dict = {}
        spam_labels = []
        # for fileName in normFileList:
        #     res = {}
        #     text = readFile(another_class+'/'+fileName)
        #     res["origin"] = text
        #     result = clean_text_ch(text, self.stopwords)
        #     res["dict"] = result[0]
        #     self.all_dict.update(res["dict"])
        #     res["list"] = result[1]
        #     res["text"] = result[2]
        #     res["tag"] = "普通内容"
        #     self.numbers.update(result[3])
        #     self.symbols.update(result[4])
        #     normal_dict[fileName] = res
        #     normal_labels.append(0)
        if extend_cur == "csv":
            csvFile = open(raw_data, "r")
            reader = csv.reader(csvFile)
            for item in reader:
                res = {}
                res["origin"] = item[0]
                result = clean_text_ch(item[0], self.stopwords)
                res["dict"] = result[0]
                self.all_dict.update(res["dict"])
                res["list"] = result[1]
                res["text"] = result[2]
                res["tag"] = item[1]
                self.numbers.update(result[3])
                self.symbols.update(result[4])
                if res["tag"] == "普通内容":
                    normal_dict[reader.line_num] = res
                    normal_labels.append(0)
                if res["tag"] == "垃圾内容":
                    spam_dict[reader.line_num] = res
                    spam_labels.append(1)

        if extend_cur == "xlsx":
            data = xlrd.open_workbook(raw_data)
            table = data.sheet_by_index(0)
            nrows = table.nrows
            for row in range(nrows):
                res = {}
                res["origin"] = table.cell(row, 0).value
                result = clean_text_ch(res["origin"], self.stopwords)
                res["dict"] = result[0]
                self.all_dict.update(res["dict"])
                res["list"] = result[1]
                res["text"] = result[2]
                self.numbers.update(result[3])
                self.symbols.update(result[4])
                res["tag"] = table.cell(row, 1).value
                if res["tag"] == "普通内容":
                    normal_dict[row] = res
                    normal_labels.append(0)
                if res["tag"] == "垃圾内容":
                    spam_dict[row] = res
                    spam_labels.append(1)

       
        # for fileName in spamFileList:
        #     res = {}
        #     text = readFile(one_class+'/'+fileName)
        #     res["origin"] = text
        #     result = clean_text_ch(text, self.stopwords)
        #     res["dict"] = result[0]
        #     self.all_dict.update(res["dict"])
        #     res["list"] = result[1]
        #     res["text"] = result[2]
        #     res["tag"] = "垃圾内容"
        #     self.numbers.update(result[3])
        #     self.symbols.update(result[4])
        #     spam_dict[fileName] = res
        #     spam_labels.append(1)

        random_seed = rm_seed
        t_data, r_data = [], []
        train_x, real_x, train_y, real_y = train_test_split(list(spam_dict.items()), spam_labels, test_size=0.2, random_state=random_seed)
        t_data.extend(train_x)
        r_data.extend(real_x)
        train_x, real_x, train_y, real_y = train_test_split(list(normal_dict.items()), normal_labels, test_size=0.2, random_state=random_seed)
        t_data.extend(train_x)
        r_data.extend(real_x)
        print("划分后训练集长度 {} 测试集长度 {}".format(len(t_data), len(r_data)))

        self.train_data, self.test_data = {}, {}
        self.train_x_dict, self.train_x_list, self.train_x_text = [], [], []
        self.train_y = []
        for (key, value) in t_data:
            self.train_data[key] = value
            self.train_x_dict.append(value["dict"])
            self.train_x_list.append(value["list"])
            self.train_x_text.append(value["text"])
            if value['tag'] == "垃圾内容":
                self.train_y.append(1)
            else:
                self.train_y.append(0)
        for (key, value) in r_data:
            self.test_data[key] = value

        self.test_data_items = self.test_data.items()

        self.test_x_dict = [v["dict"] for k, v in self.test_data_items]
        self.test_x_list = [v["list"] for k, v in self.test_data_items]
        self.test_x_text = [v["text"] for k, v in self.test_data_items]
        self.test_y = [1 if v["tag"] == "垃圾内容" else 0 for k, v in self.test_data_items]

