import torch.nn.functional as F
from torch import nn
import torch

torch_model_seq = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Flatten(), #131072
    nn.Linear(131072, 128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,4),
)

torch_model_brightness = nn.Sequential(
    nn.Linear(1,256*256),
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Flatten(), #131072
    nn.Linear(131072, 128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,4),
)


class Neural_network_LARD(nn.Module):
    def __init__(self):
        super(Neural_network_LARD, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 =nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.flatten = nn.Flatten() #131072
        self.linear7 = nn.Linear(131072, 128)
        self.linear9 = nn.Linear(128,128)
        self.linear11 = nn.Linear(128,4)
        self.relu = nn.ReLU()
   
    def forward(self, x):
        x = self.conv0(x)
        x =  self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear9(x)
        x = self.relu(x)
        x = self.linear11(x)
        return(x)


class Neural_network_LARD_BrightnessContrast(nn.Module):
    def __init__(self):
        super(Neural_network_LARD, self).__init__()
        self.linear_perturbation = nn.Linear(1,256*256) # to check
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv1 =nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.flatten = nn.Flatten() #131072
        self.linear1 = nn.Linear(131072, 128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,4)
   
    def forward(self, x):
        x = self.linear_perturbation(x)
        x = self.conv0(x)
        x =  self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)