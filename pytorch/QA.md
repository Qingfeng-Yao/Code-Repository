## 张量
- AttributeError: 'Tensor' object has no attribute 'T'
    - 改成.t()

## cuda
- 运行到model.cuda()卡死
    - torch和cuda不匹配，如可以使用torch=1.6.0和cuda=10.1