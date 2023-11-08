import os
import torch
import torch.nn as nn

class AtariDeepNet(nn.Module):
    def __init__(self, observation_space, output_size):
        super(AtariDeepNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def save_model(self, filename):
        model_folder = 'model'
        model_name = filename
        os.makedirs(model_folder, exist_ok=True)
        file_path = os.path.join(model_folder, model_name)
        torch.save(self.state_dict(), file_path)