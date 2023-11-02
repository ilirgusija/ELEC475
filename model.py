#   Partial implementation of:
#       Huang & Belongie, "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization", arXiv:1703.06868v2, 30 July 2017
#
#   Adapted from:
#       https://github.com/naoto0804/pytorch-AdaIN
#

import torch.nn as nn

class encoder_decoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),                                    # conv1-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),                                   # conv3-64
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),                                  # conv3-64
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),##########################
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),                                 # conv3-128
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),                                # conv3-128
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),###########################
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),                                # conv3-256
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),                                # conv3-256
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),                                # conv3-256
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),                                # conv3-256
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),############################
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),                                # conv3-512
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    decoder10van = nn.Sequential(
        # run some fully connected layers
        nn.Linear(8192, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    decoder100van = nn.Sequential(
        # run some fully connected layers
        nn.Linear(8192, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )
    
    
class image_classifier(nn.Module):
    def __init__(self, encoder, decoder=None):
        super(image_classifier, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # need access to these intermediate encoder steps
        # for the AdaIN computation
        encoder_list = list(encoder.children())
        self.encoder_stage_1 = nn.Sequential(*encoder_list[:4])  # input -> relu1_1
        self.encoder_stage_2 = nn.Sequential(*encoder_list[4:11])  # relu1_1 -> relu2_1
        self.encoder_stage_3 = nn.Sequential(*encoder_list[11:18])  # relu2_1 -> relu3_1
        self.encoder_stage_4 = nn.Sequential(*encoder_list[18:31])  # relu3_1 -> relu4_1

        self.decoder = decoder
        #   if no decoder loaded, then initialize with random weights
        if self.decoder == None:
            # self.decoder = _decoder
            self.decoder = encoder_decoder.decoder
            self.init_decoder_weights(mean=0.0, std=0.01)

    def init_decoder_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)
    
    def encode(self, X):
        relu1_1 = self.encoder_stage_1(X)
        relu2_1 = self.encoder_stage_2(relu1_1)
        relu3_1 = self.encoder_stage_3(relu2_1)
        relu4_1 = self.encoder_stage_4(relu3_1)
        return relu1_1, relu2_1, relu3_1, relu4_1

    def decode(self, X):
        return(self.decoder(X))

    def forward(self, X):
        _, _, _, relu4_1 = self.encode(X)
        relu4_1_flatten = relu4_1.view(relu4_1.size(0), -1)  # Flatten
        return self.decode(relu4_1_flatten)
