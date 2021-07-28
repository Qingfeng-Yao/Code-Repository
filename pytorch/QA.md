## 张量
- AttributeError: 'Tensor' object has no attribute 'T'
    - 改成.t()
- 函数中的维度索引
    - 维度索引值的轴从零开始；如果您指定轴是负数，则从最后向后进行计数，也就是倒数

## cuda
- 运行到model.cuda()卡死
    - torch和cuda不匹配，如可以使用torch=1.6.0(`torch.__version__`和`torch.version.cuda`)和cuda=10.1(`nvcc -V`)
    - `pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`