## 张量
- AttributeError: 'Tensor' object has no attribute 'T'
    - 改成.t()

## cuda
- 运行到model.cuda()卡死
    - torch和cuda不匹配，如可以使用torch=1.6.0(`torch.__version__`和`torch.version.cuda`)和cuda=10.1(`nvcc -V`)
    - `pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`