import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import random_split
class CustomConvAutoencoder(nn.Module):
    def __init__(self):
        super(CustomConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(1,2), stride=1, padding=0), 
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(size=(8, 8), mode='nearest'),  
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1, padding=0),
        )

        # self.linear3 = nn.Linear(8 * 8, 7 * 9,bias=False)
        # self.linear1 = nn.Linear(9 * 12, 7 * 9,bias=False)
        # self.linear2 = nn.Linear(7 * 9, 7 * 9,bias=False)
        
        self.linear = nn.Linear(64, 63,bias=False)
       

    def forward(self, x):
        x = self.encoder(x)
        # x = self.decoder(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear3(x)
        # x = x.view(-1, 1, 7, 9)
        return x

