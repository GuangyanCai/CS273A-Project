import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout



class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list, act: str, with_bn=False, with_do=False):
        super(MLP, self).__init__()
        num_hidden = len(hidden_sizes)
        assert(num_hidden >= 1)

        def get_block(input_size, output_size):
            if act == 'relu':
                act_fn = nn.ReLU()  
            elif act == 'tanh':
                act_fn = nn.Tanh()
            else:
                assert False
            block = [
                nn.Linear(input_size, output_size),
                act_fn
            ]
            if with_bn:
                block.insert(1, nn.BatchNorm1d(output_size))
            if with_do:
                block.append(nn.Dropout())
            return block

        self.fc = []
        self.fc += get_block(input_size, hidden_sizes[0])
        if num_hidden > 1:
            for i in range(1, num_hidden):
                self.fc += get_block(hidden_sizes[i - 1], hidden_sizes[i])
        self.fc += get_block(hidden_sizes[-1], output_size)
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)