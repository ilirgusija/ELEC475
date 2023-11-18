import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from KittiAnchors import Anchors
from classifier import object_classifier, backends

class YODA(nn.Module):
    def __init__(self, classifier_file, anchors=None):
        super(YODA, self).__init__()
        self.classifier = object_classifier(encoder=backends.encoder_se_resnet)
        classifier_state_dict = torch.load(classifier_file)
        self.classifier.load_state_dict(classifier_state_dict)
        if anchors==None:
            anchors = Anchors()

    def forward(self, X):
        # Assuming your classifier's forward method can handle the input directly
        return self.classifier(X)
