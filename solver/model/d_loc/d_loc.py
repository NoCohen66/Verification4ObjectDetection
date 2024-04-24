import torch 
import torch.nn.functional as F
from torch import nn

class NeuralNetwork_OL_v2(nn.Module):
    '''
    New convolutional model (v2)
    '''
    def __init__(self, classif=True):
        super(NeuralNetwork_OL_v2, self).__init__()
        
        self.classif=classif
        seed = 0
        torch.manual_seed(seed)
        padding = 1
        
        self.conv0 = nn.Conv2d(1, 16, 3, padding= padding) # 3x3 filters w/ same padding
        self.pool0 = nn.MaxPool2d(2, stride=2)
        #output shape : 15x15x16
        self.conv1 = nn.Conv2d(16, 16, 3, padding= padding) # 3x3 filters w/ same padding
        
        #print("before self.conv1.weight\n",a)

        
        # Initializing the weights with the Xavier initialization method (equivalent to tf.keras.Initializers.glorot_uniform)
        #nn.init.xavier_uniform(self.conv1.weight)
        #print("after self.conv1.weight\n", b)

        #output shape : 17x17x16 ((15 - 3 + 2*2)/1 + 1 )
         #initialize to zeros 
        #nn.init.zeros_(self.conv1.weight)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        #output shape : 8x8x16
        self.flatten = nn.Flatten()
        #output shape : 1024
        #HERE CHECK RIGHT SIZE FROM FLATTEN TO LINEAR

        self.linear_relu_stack= nn.Linear(7744,256)
        if self.classif:
            self.linear = nn.Linear(256, 10)
        else:
            self.linear_all = nn.Linear(256, 4) 
        """
        self.linear_x = nn.Linear(256, 1)
        self.linear_y = nn.Linear(256, 1)
        self.linear_all = nn.Linear(256, 2) 
        """
        #modifiy here, logits layer must return label, coordinates xmin,ymin,xmax,ymax
        #self.linear_all = nn.Linear(256, 4) 
        
        
    
    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.pool0(x))
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x= F.relu(x)
        if self.classif:
            logits = self.linear(x)
        else:
            logits = self.linear_all(x)

        return logits#logits, x_min, y_min, x_max, y_max
    
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
