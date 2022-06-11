import jieba

def clean_chinese_text(text):
    seg_list = jieba.cut(text.strip())
    re_list = []
    for word in seg_list:
        word = word.strip()
        if word == "":
            continue
        re_list.append(word)
    return ' '.join(re_list)