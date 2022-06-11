import scipy.io as scio
import torch.nn as nn
import torch

import models

# optional params
data_option = "train_2" # ["train_1", "train_2"]
model_option = "three_linear" # ["one_linear", "three_linear"]
reverse = True
# ---------------

data_path = "data/"+data_option
data_x_name = data_path+"/X.mat"
data_y_name = data_path+"/Y.mat"

data_x = scio.loadmat(data_x_name) # dict
data_y = scio.loadmat(data_y_name) # dict
if data_option == "train_1":
    # print(type(data_x["X"])) # <class 'numpy.ndarray'>
    # print(type(data_y["Y"])) # <class 'numpy.ndarray'>
    # print(data_x["X"].shape) # (100000, 18)
    # print(data_y["Y"].shape) # (100000, 121)
    if reverse:
        x_train = data_y["Y"]
        y_train = data_x["X"]
    else:
        x_train = data_x["X"]
        y_train = data_y["Y"]

elif data_option == "train_2":
    # print(type(data_x["X2"])) # <class 'numpy.ndarray'>
    # print(type(data_y["Y2"])) # <class 'numpy.ndarray'>
    # print(data_x["X2"].shape) # (309983, 18)
    # print(data_y["Y2"].shape) # (309983, 121)
    if reverse:
        x_train = data_y["Y2"]
        y_train = data_x["X2"]
    else:
        x_train = data_x["X2"]
        y_train = data_y["Y2"]

# Hyper-parameters
if reverse:
    input_size = 121
    output_size = 18
else:
    input_size = 18
    output_size = 121
num_epochs = 60
learning_rate = 0.001

# Linear regression model
if model_option == "one_linear":    
    model = nn.Linear(input_size, output_size)
elif model_option == "three_linear":
    model = models.Multilayers(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

# Train the model
for epoch in range(num_epochs):
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
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

