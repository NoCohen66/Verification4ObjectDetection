import torch.nn.functional as F
from torch import nn
import torch

class NeuralNetwork_BrightnessContrast(nn.Module):
    '''
    New model that takes as input a brightness or constrat value and apply it to a specific image
    using the linear_perturbation layer
    (aka the weight and biases of the linear_perturbation layer are set to specific values to encode
    brightness or contrast for a specific image)
    '''
    def __init__(self, classif=True):
        super(NeuralNetwork_BrightnessContrast, self).__init__()
        self.classif=classif
        seed = 0
        torch.manual_seed(seed)
        padding = 1
        self.linear_perturbation = nn.Linear(1,90*90)
        self.conv0 = nn.Conv2d(1, 16, 3, padding= padding) # 3x3 filters w/ same padding
        self.pool0 = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(16, 16, 3, padding= padding) # 3x3 filters w/ same padding
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack= nn.Linear(7744,256)
        if self.classif:
            self.linear = nn.Linear(256, 10)
        else:
            self.linear_all = nn.Linear(256, 4) 

    
    def forward(self, alpha):
        layer = self.linear_perturbation(alpha)
        layer = layer.view((-1, 1, 90, 90))
        layer = self.conv0(layer)
        layer = F.relu(self.pool0(layer))
        layer = self.conv1(layer)
        layer = F.relu(self.pool1(layer))
        layer = self.flatten(layer)
        layer = self.linear_relu_stack(layer)
        layer = F.relu(layer)
        if self.classif:
            logits = self.linear(layer)
        else:
            logits = self.linear_all(layer)
        return logits
