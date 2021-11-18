import os
import json
from bs4 import BeautifulSoup
import re

raw_path = "raw_data"
input_path = "input_data"

if not os.path.exists(input_path):
        os.makedirs(input_path)

input_dict = {}
for root, dirs, files in os.walk(raw_path):
        for f in files:
                if f[-4:] == 'json':
                        file_path = root +'/'+ f
                        input_dict[f.split('.')[0]] = {}
                        with open(file_path,'r',encoding='utf-8') as load_f:
                                load_dict = json.load(load_f)
                                title = load_dict["title"]
                                input_dict[f.split('.')[0]]["title"] = title

                                assert type(load_dict["content"]) in [str]
                                soup = BeautifulSoup(load_dict["content"])
                                for t in soup(['style', 'script']):
                                        t.extract()
                                text = soup.get_text()
                                text = re.sub('\u3000', ' ', text)
                                lines = [' '.join(line.split()) for line in text.splitlines()]
                                text = '\n'.join([line for line in lines if line])
                                input_dict[f.split('.')[0]]["content"] = text
print("total documents: {}".format(len(input_dict)))
with open(input_path+'/data.json','w',encoding='utf-8') as w_f:
        json.dump(input_dict, w_f, indent=4)
                                
