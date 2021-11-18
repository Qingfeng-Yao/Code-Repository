import os
import json
from bs4 import BeautifulSoup
import re

raw_path = "raw_data"
chinese_path = "chinese_data"

num_jsonfiles = 0
uni_jsonfiles = set()

content_len = 0
for root, dirs, files in os.walk(raw_path):
        if len(dirs) != 0:
                print("raw_data has {} classes".format(len(dirs)))
        for f in files:
                if f[-4:] == 'json':
                        num_jsonfiles += 1
                        uni_jsonfiles.add(f)

                        file_path = root +'/'+ f
                        sub_dir = root.split('/')[-1]
                        new_path = chinese_path+'/'+sub_dir
                        if not os.path.exists(new_path):
                                os.makedirs(new_path)
                        with open(file_path,'r',encoding='utf-8') as load_f:
                                load_dict = json.load(load_f)
                                assert type(load_dict["content"]) in [str]

                                soup = BeautifulSoup(load_dict["content"])
                                for t in soup(['style', 'script']):
                                        t.extract()
                                text = soup.get_text()

                                text = re.sub('\u3000', ' ', text)
                                lines = [' '.join(line.split()) for line in text.splitlines()]
                                text = '\n'.join([line for line in lines if line])

                                text = load_dict["title"] + '\n' + text
                                content_len += len(text)

                                with open(new_path+'/'+f.split('.')[0]+'.txt','w',encoding='utf-8') as w_f:
                                                w_f.write(text)

                        
assert num_jsonfiles == len(uni_jsonfiles)
print("num json files: {}".format(num_jsonfiles))
print("ave file content(including title) len: {}".format(content_len/num_jsonfiles))