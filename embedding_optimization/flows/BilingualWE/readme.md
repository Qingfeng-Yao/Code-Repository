## 参考资料
- 2019 | ACL | Density matching for bilingual word embedding | Chunting Zhou et al.
- [violet-zct/DeMa-BWE](https://github.com/violet-zct/DeMa-BWE)

## 程序运行
- 环境配置: `python3.6`; `requirements.txt`
- 数据准备: 提前下载好存放在data文件夹下
    - fasttext: 存放src_emb或tgt_emb，如wiki.en.bin。下载地址: `https://fasttext.cc/docs/en/crawl-vectors.html#models`
    - dictionaries: 存放训练或测试字典，如en-zh.0-5000.txt/en-zh.5000-6500.txt。下载地址: `https://github.com/facebookresearch/MUSE`
    - monolingual: 存放evaluation datasets，该文件夹下包括多个子文件，如`en`。下载地址: `https://dl.fbaipublicfiles.com/arrival/wordsim.tar.gz`
    - crosslingual: 存放双语evaluation datasets，如`en-de-SEMEVAL17.txt`。下载地址: `http://alt.qcri.org/semeval2017/task2/data/uploads/semeval17_task2_test.zip`。同时存放Europarl语料库，如`europarl-v7.en-de.en`。下载地址: `http://www.statmt.org/europarl/v7/europarl.tgz`
- 执行命令: 
    - 先创建目录`saved_exps`
    - 注意应导入`import fasttext`
    - `en2zh`: `python3 main_e2e.py --model_name id_en_zh --export_emb 1 --supervise_id 6000 --valid_option unsup --src_train_most_frequent 20000 --tgt_train_most_frequent 20000 --src_base_batch_size 20000 --tgt_base_batch_size 20000 --batch_size 2048 --sup_s_weight 5. --sup_t_weight 5. --s_var 0.01 --s2t_t_var 0.015 --t_var 0.015 --t2s_s_var 0.02 --src_emb_path data/fasttext/wiki.en.bin --tgt_emb_path data/fasttext/wiki.zh.bin --n_steps 150000 --display_steps 100 --valid_steps 5000 --sup_dict_path data/dictionaries/en-zh.0-5000.txt --dico_eval data/dictionaries/en-zh.5000-6500.txt > log.txt`