import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1=nn.Conv2d(1,8,kernel_size=5,stride=2,padding=2)
        self.relu1=nn.ReLU()
        self.bn1=nn.BatchNorm2d(8)
        init.kaiming_normal(self.conv1.weight,a=0.1)
        self.conv1.bias.data.zero_()
        self.conv_layers1=nn.Sequential(self.conv1,self.relu1,self.bn1)

        self.conv2=nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(16)
        self.conv2.bias.data.zero_()
        self.relu2 = nn.ReLU()
        self.conv_layers2 =nn.Sequential(self.conv2, self.relu2, self.bn2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal(self.conv1.weight, a=0.1)
        self.conv3.bias.data.zero_()
        self.conv_layers3 = nn.Sequential(self.conv3, self.relu3, self.bn3)



        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=32, out_features=3)
        self.conv_layers=[self.conv_layers1 ,self.conv_layers2, self.conv_layers3]
        self.Conv=nn.Sequential(*self.conv_layers)

    def forward(self,x):
        x=self.Conv(x)
        x=self.ap(x)

        x=x.view(x.shape[0],-1)
        x = self.lin(x)
        return x


