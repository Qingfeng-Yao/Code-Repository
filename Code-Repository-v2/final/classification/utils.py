import argparse
import jieba
import os

from torchtext import data
from torchtext.vocab import Vectors

from sklearn.model_selection import train_test_split

def clean_chinese_text(text, stopwords):
    seg_list = jieba.cut(text.strip())
    re_list = []
    for word in seg_list:
        word = word.strip()
        if word not in stopwords:
            if word != "\t" and word != "\n":
                re_list.append(word)
    return ' '.join(re_list)

def readStopwords(path):
    stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]
    return stopwords

def readFile(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def train_and_test_split(imb_data, args, stopwords, label_dict):
    dict_text2label = {}
    for k, v in imb_data.items():
        if v["label"] not in dict_text2label:
            dict_text2label[v["label"]] = [v["content"]]
        else:
            dict_text2label[v["label"]].append(v["content"])
    print("classes {}".format(len(dict_text2label)))

    train_data = {}
    test_data = {}
    val_data = {}
    id_doc = 0
    for k, v in dict_text2label.items():
        train_x, test_x, train_y, test_y = train_test_split(v, [k for _ in range(len(v))], test_size=args.test_size, random_state=args.seed)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=args.seed)
        for t in train_x:
            train_data[id_doc] = {"text":clean_chinese_text(t, stopwords), "tag":k}
            id_doc += 1
        for t in test_x:
            test_data[id_doc] = {"text":clean_chinese_text(t, stopwords), "tag":k}
            id_doc += 1
        for t in val_x:
            val_data[id_doc] = {"text":clean_chinese_text(t, stopwords), "tag":k}
            id_doc += 1
    print(len(train_data), len(test_data), len(val_data))

    testing_items = test_data.items()
    data_test = [v["text"] for k, v in testing_items]
    test_label = [label_dict[v["tag"]] for k, v in testing_items]
    training_items = train_data.items()
    data_train = [v["text"] for k, v in training_items]
    train_label = [label_dict[v["tag"]] for k, v in training_items]
    val_items = val_data.items()
    data_val = [v["text"] for k, v in val_items]
    val_label = [label_dict[v["tag"]] for k, v in val_items]

    return data_test, test_label, data_train, train_label, data_val, val_label

def word_cut(text):
    return [word.strip() for word in text.split()]

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors

def load_cnn_dataset(path, text_field, label_field, args, **kwargs):
    text_field.tokenize = word_cut
    train_dataset, dev_dataset, test_dataset = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv', test='test.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset, test_dataset)
    label_field.build_vocab(train_dataset, dev_dataset, test_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset), len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter, test_iter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='set random seed', type=int, default=123)
    # base_model
    parser.add_argument('--model_name', help='options: nb | lr', type=str, default='nb')
    parser.add_argument('--test_size', type=float, default=0.4)
    # text_cnn
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('--embedding_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('--filter_num', type=int, default=100, help='number of each size of filter')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='comma-separated filter sizes to use for convolution')
    parser.add_argument('--static', type=bool, default=True, help='whether to use static pre-trained word vectors')
    parser.add_argument('--non_static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
    parser.add_argument('--multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
    parser.add_argument('--pretrained_name', type=str, default='sgns.zhihu.word', help='filename of pre-trained word vectors')
    parser.add_argument('--pretrained_path', type=str, default='static', help='path of pre-trained word vectors')
    parser.add_argument('--device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('--test_interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('--early_stopping', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    # pretrain_moco

    return parser.parse_args()
