import os
import torch
import torch.nn as nn

class QNetModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function=nn.ReLU()):  
        super().__init__()              
        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size))
            self.layers.append(activation_function)
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save_model(self):
        model_folder = 'model'
        model_name = 'q_net.pth'
        os.makedirs(model_folder, exist_ok=True)
        file_path = os.path.join(model_folder, model_name)
        torch.save(self.state_dict(), file_path)
