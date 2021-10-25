## 张量
- AttributeError: 'Tensor' object has no attribute 'T'
    - 改成.t()
- 函数中的维度索引
    - 维度索引值的轴从零开始；如果您指定轴是负数，则从最后向后进行计数，也就是倒数

## cuda
- 运行到model.cuda()卡死
    - torch和cuda不匹配，如可以使用torch=1.6.0(`torch.__version__`和`torch.version.cuda`)和cuda=10.1(`nvcc -V`)
    - `pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
- 出现错误`RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED`，显存不足导致

## Bert
- from pytorch_pretrained_bert.modeling import BertModel; BertModel.from_pretrained函数中的名字参数已经改为`pretrained_model_name_or_path`
- 下载嵌入时可能会报错`Model name 'bert-base-uncased' was not found in model name list`，再重新运行一下就好
    - 如果重复多次仍旧报错，可将bert-base-uncased改称路径名称，该路径下存放`bert_config.json`、`pytorch_model.bin`以及`vocab.txt`文件。`vocab.txt`可根据报错信息中提供的链接进行下载，`json`和`model.bin`来自压缩文件`bert-xxx-xxx.tar.gz`，下载网址如`https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz`
- 要使用上述bert嵌入时，也应用bert进行分词，如下函数所示
- from pytorch_pretrained_bert import BertTokenizer; 继承BertTokenizer自定义函数时会报错`TypeError: __init__() got an unexpected keyword argument 'max_len'`，出错的地方在于`tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)`；修改的思路则是在定义函数的init函数中添加参数max_len

## Repo
- 安装torchnlp应使用pytorch-nlp
- torch版本太低如1.0.0，则无法使用torch.optim.AdamW

## Training
- 如果随着epoch增加，auc一直在下降，可以调小学习率