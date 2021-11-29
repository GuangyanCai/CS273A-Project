import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list, device='cuda'):
        super(MLP, self).__init__()
        num_hidden = len(hidden_sizes)
        assert(num_hidden >= 1)
        self.fc = nn.ModuleList()
        self.fc.extend([nn.Linear(input_size, hidden_sizes[0], device=device)])
        if num_hidden > 1:
            for i in range(1, num_hidden):
                self.fc.extend([nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], device=device)])
                self.fc.extend([nn.Dropout(p=0.5)])
        self.fc.extend([nn.Linear(hidden_sizes[-1], output_size, device=device)])
        self.relu = F.relu

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = self.relu(x)

        return x