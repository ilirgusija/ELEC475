import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init
from torchvision.models import ResNet18_Weights

class CustomKeypointModel(nn.Module):
    def __init__(self):
        super(CustomKeypointModel, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

        fc_input = resnet18.fc.in_features

        self.upconv1 = nn.ConvTranspose2d(fc_input, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self._initialize_upsampling_weights()

    def _initialize_upsampling_weights(self):
        for m in [self.upconv1, self.upconv2, self.upconv3, self.upconv4]:
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet18(x)

        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.upconv4(x))
        x = self.relu(self.upconv5(x))


        x = self.sigmoid(x)

        return x

class single_point(nn.Module):
    def __init__(self):
          super(single_point, self).__init__()
          resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

