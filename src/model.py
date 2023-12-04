import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init
from torchvision.models import ResNet18_Weights

import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init


class CustomKeypointModel(nn.Module):
    def __init__(self):
        super(CustomKeypointModel, self).__init__()
        resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

        fc_input = resnet18.fc.in_features

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(fc_input, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet18(x)

        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.upsample(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.upsample(x)
        x = self.sigmoid(self.conv5(x))

        return x


class single_point(nn.Module):
    def __init__(self):
          super(single_point, self).__init__()
          resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
          self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
          fc_input = resnet18.fc.in_features
          self.fc = nn.Linear(fc_input, 2)
          self.sigmoid = nn.Sigmoid()
          self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc]:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

