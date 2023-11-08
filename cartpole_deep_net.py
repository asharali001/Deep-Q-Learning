import os
import torch
import torch.nn as nn

class CartpoleDeepNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):  
        super().__init__()              
        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size))
            self.layers.append(nn.ReLU())
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save_model(self, filename):
        model_folder = 'model'
        model_name = filename
        os.makedirs(model_folder, exist_ok=True)
        file_path = os.path.join(model_folder, model_name)
        torch.save(self.state_dict(), file_path)