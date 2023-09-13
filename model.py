from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms, datasets

# step 2


idx = 1

# plt.imshow(train_set.data[idx], cmap='gray')
# plt.show()

# step 3
class MLP(nn.Module):
    
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(MLP, self).__init__()
        N2=392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck,N2)
        self.fc4 = nn.Linear(N2, N_output)
        self.type = 'MLP4'
        self.input_type = (1, 28*28)
        
    def forward(self, X):
        # encoder
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        
        # decoder
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = F.sigmoid(X)
        
        return X
        
        
        
        
    
    
    
