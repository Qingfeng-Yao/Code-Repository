import scipy.io as scio
import torch.nn as nn
import torch
import pandas as pd
import os
import shutil
import numpy as np

import models

reverse = False

## 数据集设置
data_path = "data/train_2"
out_path = "out/"
data_x_name = data_path+"/X.mat"
data_y_name = data_path+"/Y.mat"

data_x = scio.loadmat(data_x_name) 
data_y = scio.loadmat(data_y_name) 

if reverse:
    x = data_y["Y2"]
    y = data_x["X2"]
else:
    x = data_x["X2"]
    y = data_y["Y2"]

print(x.shape, y.shape)

x_data = x[:300000]
y_data = y[:300000]
train_num = int(0.7*300000)
o_x_train = x_data[:train_num]
o_y_train = y_data[:train_num]
shuffle_index = np.arange(len(o_x_train))
np.random.shuffle(shuffle_index)
x_train = o_x_train[shuffle_index]
y_train = o_y_train[shuffle_index]
x_val = x_data[train_num:]
y_val = y_data[train_num:]
x_test = x[300000:]
y_test = y[300000:]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

if reverse:
    input_size = 121
    output_size = 18
else:
    input_size = 18
    output_size = 121
num_epochs = 100
learning_rate = 0.001

## 模型创建
# model = models.Multilayers(input_size, output_size)
model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

## 模型训练: 验证集/保存最佳模型; 调整学习率
best_loss = 100000
def val(x_val, y_val, model):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(x_val).to(torch.float32)
        targets = torch.from_numpy(y_val).to(torch.float32)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

    return loss.item()

def save_checkpoint(state, is_best):
    filename = 'checkpoints/ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 80:
        lr = lr * 0.0001
    elif epoch > 60:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints") 

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, learning_rate)
    model.train()
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train).to(torch.float32)
    targets = torch.from_numpy(y_train).to(torch.float32)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    val_loss = val(x_val, y_val, model)
    
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)
    output_best = 'Best Val Loss: %.3f' % best_loss
    print(output_best)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if not os.path.exists("out"):
    os.makedirs("out") 

## 模型预测
resume = "checkpoints/ckpt.best.pth.tar"
print("===> Loading checkpoint {}".format(resume))
checkpoint = torch.load(resume)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)

predicted = model(torch.from_numpy(x_test).to(torch.float32)).detach().numpy()
real = y_test
predicted = pd.DataFrame(predicted)
real = pd.DataFrame(real)
if reverse:
    writer_1 = pd.ExcelWriter(out_path+'pred_X.xlsx')
    writer_2 = pd.ExcelWriter(out_path+'real_X.xlsx')
else:
    writer_1 = pd.ExcelWriter(out_path+'pred_Y.xlsx')
    writer_2 = pd.ExcelWriter(out_path+'real_Y.xlsx')
real.to_excel(writer_2, 'sheet_1', float_format='%.2f')
writer_2.save()
writer_2.close()
predicted.to_excel(writer_1, 'sheet_1', float_format='%.2f')
writer_1.save()
writer_1.close()

