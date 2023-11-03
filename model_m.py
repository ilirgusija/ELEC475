import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def downsample_layer(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class encoder_decoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),  # This remains unchanged
        nn.ReflectionPad2d((1, 1, 1, 1)),
        ResidualBlock(3, 64, downsample=downsample_layer(3, 64, 1)),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        ResidualBlock(64, 128, downsample=downsample_layer(64, 128, 1)),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        ResidualBlock(128, 256, downsample=downsample_layer(128, 256, 1)),
        ResidualBlock(256, 256),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        ResidualBlock(256, 512, downsample=downsample_layer(256, 512, 1)),
    )

class mod_NN(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_classes=10):
        super(mod_NN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder == None:
            self.encoder = encoder_decoder.encoder
            self.init_encoder_weights(mean=0.0, std=0.01)
        if self.decoder == None:
            self.decoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(512, num_classes),
            )
            self.init_decoder_weights(mean=0.0, std=0.01)
            
    def init_encoder_weights(self, mean, std):
        for param in self.encoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)
    
    def init_decoder_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        return self.decode(self.encode(X))
