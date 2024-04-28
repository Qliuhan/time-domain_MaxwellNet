import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

from loss.maxwell import Maxwell2DMur
from architecture.PINN import Sequentialmodel
from model.model import Model

np.random.seed(123)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    c0 = 1/np.sqrt(args.e0*args.u0)
    dt =0.5/(c0*np.sqrt(1/(args.dx**2)+1/(args.dy**2)))
    x_max = args.dx*args.Nx
    y_max = args.dy*args.Ny
    t_max = dt*args.steps  

    x_ = np.linspace(start=0, stop=x_max, num=args.Nx, endpoint=False)  
    y_ = np.linspace(start=0, stop=y_max, num=args.Ny, endpoint=False) 
    t_ = np.linspace(start=0, stop=t_max, num=args.steps, endpoint=False)  
    X, T, Y = np.meshgrid(x_, t_, y_)  

    # network
    input_width = args.input_width
    layer_width = args.layer_width
    network = Sequentialmodel(input_width=input_width, layer_width=layer_width)

    if torch.cuda.device_count() > 1:
        print("==> Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network)
    network.to(device)


    
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')

    Maxwell = Maxwell2DMur(criterion=criterion, args=args)
    model = Model(network=network, 
                  optimizer=optimizer, 
                  Maxwell=Maxwell, 
                  criterion=criterion,
                  args=args
                  )                       

    model.train()  


if __name__ == '__main__':
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

    parser = argparse.ArgumentParser()

    parser.add_argument('-e0', help='The value of the dielectric constant in vacuum', default=1.0)
    parser.add_argument('-u0', help='The value of magnetic permeability in vacuum', default=1.0)
    parser.add_argument('-dx', help='dx', default=0.05)
    parser.add_argument('-dy', help='dy', default=0.05)
    parser.add_argument('-Nx', type=int, default=100)
    parser.add_argument('-Ny', type=int, default=100)
    parser.add_argument('-steps', type=int, default=100)
    parser.add_argument('-input_width', type=int, default=7)
    parser.add_argument('-layer_width', type=int, default=200)
    parser.add_argument('-lr', help='learning rate', default=0.0001)
    parser.add_argument('-num_items', type=int, default=100001)
    parser.add_argument('-path', help='Path to save model parameters', default='path')
    parser.add_argument('-sampling_size', help='Number of sampling points in one training step', type=int, default='1000')
    parser.add_argument('-init_path', help='Path to load initial field', default='init_field.mat')

    args = parser.parse_args()

    main(args) 


