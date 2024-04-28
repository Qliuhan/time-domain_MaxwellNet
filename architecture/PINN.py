import torch
from torch import nn
import numpy as np

lb = np.array([0., 0., 0., 2.5, 0.5, 2.0, 2.5])
ub = np.array([5, 5, 1.767766952966369, 3.5, 1.5, 3.0, 3.5]) 

class mySin(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.sin()
        return x

class Sequentialmodel(nn.Module):

    def __init__(self, input_width, layer_width):
        super().__init__()  

        'activation function'
        self.activation = mySin()

        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')

        self.iter = 0

        
        self.linear_in = torch.nn.Linear(input_width, layer_width)
        self.linear1 = torch.nn.Linear(layer_width, layer_width)
        self.linear2 = torch.nn.Linear(layer_width, layer_width)
        self.linear3 = torch.nn.Linear(layer_width, layer_width)
        self.linear4 = torch.nn.Linear(layer_width, layer_width)
        self.linear5 = torch.nn.Linear(layer_width, layer_width)
        self.linear6 = torch.nn.Linear(layer_width, layer_width)
        self.linear7 = torch.nn.Linear(layer_width, layer_width)
        self.linear8 = torch.nn.Linear(layer_width, layer_width)
        self.linear_out = torch.nn.Linear(layer_width, 3)


    'foward pass'
    def forward(self, x, para_numpy):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        input_min = torch.from_numpy(lb).float().cuda()
        input_max = torch.from_numpy(ub).float().cuda()
        

        # convert to float
        x = x.float()

            
        a = torch.cat((x, para_numpy), 1)  

        
        
        a = 2 * ((a - input_min) / (input_max - input_min)) - 1  # feature scaling

        
        y = self.linear_in(a)  # fully connect layer
        y = y + self.activation(self.linear2(self.activation(self.linear1(y))))  # residual block 1
        y = y + self.activation(self.linear4(self.activation(self.linear3(y))))  # residual block 2
        y = y + self.activation(self.linear6(self.activation(self.linear5(y))))  # residual block 3
        y = y + self.activation(self.linear8(self.activation(self.linear7(y))))  # residual block 3
        output = self.linear_out(y)

        return output