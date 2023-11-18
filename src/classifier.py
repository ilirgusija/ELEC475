import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.reduced_channel_size = in_channels // reduction_ratio
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
            nn.Linear(in_features=in_channels,
                      out_features=self.reduced_channel_size),
            nn.ReLU(),
            nn.Linear(in_features=self.reduced_channel_size,
                      out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.block(X)
        return X

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)

        return X

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, mid_channels=None):
        super(BottleneckBlock, self).__init__()
        if mid_channels == None:
            mid_channels = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        return X
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block=None):
        super(ResidualBlock, self).__init__()
        stride = 2 if in_channels != out_channels else 1
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 else None
        self.block = BasicBlock(in_channels, out_channels, stride) if block == None else block

    def forward(self, x):
        residual = x
        out = self.block.forward(x)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = F.relu(out)
        return out

class ResNeXtBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, cardinality=32):
        super(ResNeXtBlock, self).__init__(in_channels,
                                           out_channels,
                                           BottleneckBlock(in_channels=self.in_channels,
                                                            mid_channels=self.in_channels//2,
                                                            out_channels=self.out_channels,
                                                            groups=in_channels // cardinality,
                                                            stride=in_channels // out_channels),
                                           in_channels//2)

    def forward(self, x):
        residual = x
        out = self.block.forward(x)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = F.relu(out)
        return out

class SE_ResModule(nn.Module):
    def __init__(self, block, reduction_ratio=16):
        super(SE_ResModule, self).__init__()
        self.block = block
        self.se_block = SqueezeExcitationBlock(self.out_channels, reduction_ratio)

    def forward(self, X):
        residual = X
        
        res_out = self.block.forward(self, X)
        se_out = self.se_block.forward(res_out)
        
        out = res_out * se_out
        
        if self.block.downsample is not None:
            residual = self.block.downsample(X)
        
        out += residual
        out = F.relu(out)
        return out

class encoder_decoder:
    encoder_se_resnet = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        SE_ResModule(ResidualBlock(3, 64)),
        SE_ResModule(ResidualBlock(64, 128)),
        SE_ResModule(ResidualBlock(128, 256)),
        SE_ResModule(ResidualBlock(256, 256)),
        SE_ResModule(ResidualBlock(256, 512)),
    )
    encoder_se_resneXt = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        SE_ResModule(ResNeXtBlock(3, 64)),
        SE_ResModule(ResNeXtBlock(64, 128)),
        SE_ResModule(ResNeXtBlock(128, 256)),
        SE_ResModule(ResNeXtBlock(256, 256)),
        SE_ResModule(ResNeXtBlock(256, 512)),
    )

class object_classifier(nn.Module):
    def __init__(self, encoder=None, decoderFile=None, num_classes=2):
        super(object_classifier, self).__init__()
        self.encoder = encoder_decoder.encoder
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        
        if self.decoderFile!=None:
            self.decoder.load_state_dict(torch.load(decoderFile))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        return self.decode(self.encode(X))
