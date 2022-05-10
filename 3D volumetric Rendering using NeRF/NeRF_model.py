import torch
from collections import defaultdict
import torch.nn as nn
import json
import numpy as np
import math
from PIL import Image

class NeRF(nn.Module):
    
    '''
    NeRF Model as described in the paper, which is composed of 8 layers as shown in fig 7 of original paper.
    Inputs:
            num_layers : number of layers of the MLP : default 8
            num_units : number of units in the each hidden layer : default 256
            xyz_coordinates : dimension of encoded viewpoint : default 10
            dir_coordinates : dimension of encoded pose : default 4
            skip_layer : skip layer or residual layer : default 4

    Outputs:
            Returns the predicted value of volumetric density and RGB color value for each input tensor.
    '''
    
    def __init__(self, num_layers = 8, num_units = 256, xyz_coordinates = 10,
                dir_coordinates = 4, skip_layers = 4):
        
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.xyz_coordinates = xyz_coordinates
        self.dir_coordinates = dir_coordinates
        self.xyz_shape = 3 + 2 * 3 * xyz_coordinates
        self.dir_coordinates = 3 + 2 * 3 * dir_coordinates
                
        self.linear_layers = nn.ModuleList([nn.Linear(self.xyz_shape, self.num_units)] + [nn.Linear(self.num_units + self.xyz_shape, self.num_units) if (idx % skip_layers == 0) else nn.Linear(self.num_units, self.num_units) for idx in range(1, self.num_layers)])
        self.color_prediction_branch = nn.Linear(self.num_units, self.num_units)
        self.density = nn.Linear(self.num_units, 1)
        self.rgb_branch = nn.Sequential(nn.Linear(self.dir_coordinates + self.num_units, self.num_units // 2),
                                       nn.ReLU(True))
        self.rgb = nn.Sequential(nn.Linear(self.num_units // 2, 3),
                                nn.Sigmoid())
        
        
    
    def forward(self, input_coordinates, input_view_dir):
        
        input_xyz = input_coordinates
        for idx, _ in enumerate(self.linear_layers):
            if idx % 4 == 0 and idx > 0:
                input_xyz = torch.cat([input_coordinates, input_xyz], -1)
            input_xyz = self.linear_layers[idx](input_xyz)
            
        sigma = self.density(input_xyz)
        color_prediction_input = self.color_prediction_branch(input_xyz)
        view_concated_input = torch.cat([color_prediction_input, input_view_dir], -1)
        rgb_branch_output = self.rgb_branch(view_concated_input)
        rgb = self.rgb(rgb_branch_output)
        final_output = torch.cat([sigma, rgb], -1)
        return final_output
        
        

class positional_encoding(nn.Module):

    '''
    Converts the input 5D vector x to the (x, sin(2^k x), cos(2^k x), ...)
    Inputs :
            num_in_channels = number of input channels : default 3.
            num_freq_cycles = number of frequencies or encoding dimension.
    Outputs :
            Returns a embedded tensor.
    '''
    
    def __init__(self, num_in_channels, num_freq_cycles):
        super(positional_encoding, self).__init__()
        self.num_freq_cycles = num_freq_cycles
        self.num_in_channels = num_in_channels
        self.freq_range = 2**torch.linspace(0, self.num_freq_cycles - 1, self.num_freq_cycles)
        #self.freq_range = torch.linspace(1, 2**(self.num_freq_range - 1), self.num_freq_cycles)
        self.funcs = [torch.sin, torch.cos]
        
    def forward(self, x):
        
        gamma = [x]
        for f in self.freq_range:
            for func in self.funcs:
                gamma += [func( f * x)]
                
        if len(gamma) > 1:
            return torch.cat(gamma, dim = -1)
        else:
            return gamma[0]