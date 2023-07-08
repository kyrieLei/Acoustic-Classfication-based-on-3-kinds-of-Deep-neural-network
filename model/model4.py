import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.conv=nn.Conv2d(
            1,
            3,
            kernel_size=1,
            stride=1,
            padding=(0, 2),
            dilation=(1, 2),
            bias=False,
        )
        self.model = models.resnet50(pretrained=False)
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        numFit = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(numFit, 3), nn.Softmax(dim=1))

    def forward(self, x):
        x=self.conv(x)
        x=self.model(x)

        # x = x.view(x.size(0), x.size(1))
        return x


