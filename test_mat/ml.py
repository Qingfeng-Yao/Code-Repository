import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

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

x_train = x[:300000]
y_train = y[:300000]
# x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)
x_test = x[300000:]
y_test = y[300000:]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# reg = LinearRegression()
# reg.fit(x_train, y_train)

if not os.path.exists("out"):
    os.makedirs("out") 

predicted = reg.predict(x_test)
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