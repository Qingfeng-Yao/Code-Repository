import torch.nn as nn

class Multilayers(nn.Module):
    def __init__(self, input_size, output_size):
        super(Multilayers, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 360)
        self.fc2 = nn.Linear(360, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, self.output_size)

    def forward(self, x):
        out = self.fc4(nn.ReLU()(self.fc3(nn.ReLU()(self.fc2(nn.ReLU()(self.fc1(x)))))))
        return out