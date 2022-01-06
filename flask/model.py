import torch
import torch.nn as nn 

#Building the Model
# Creating a CNN class
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(166464, 1042)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1042, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.batchnorm1(x)
        x = self.max_pool1(x)
        
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.batchnorm2(x)
        x = self.max_pool2(x)
                
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        output = self.fc4(x)

        return output