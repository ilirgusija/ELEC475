import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from model import MLP
import argparse
from torchvision import transforms, datasets
from torchsummary import summary
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable


# -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
# -z=bottleneck -e=epochs -b=batch_size -p=path

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training ...')
    model.to(device)
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(target)
            data, target = data.to((device)), target.to(device)
            data = data.view(data.size(0), -1) # Flatten the data

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        scheduler.step()

        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))


def main(z, e, b, s, p):

    train_loader = DataLoader(datasets.MNIST('./data/mnist',
                                             train=True,
                                             download=True,
                                             transform=transforms.Compose([transforms.ToTensor()])),
                              batch_size=1024,
                              shuffle=True)

    model = MLP(N_bottleneck=z)
    lr = 0.001
    opt = optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    sched = StepLR(opt, step_size=100, gamma=0.1)
    train(e, opt, model, loss, train_loader, sched, 'cuda')
    summary(model, (1, 28*28))


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Training script")

    # Add arguments
    parser.add_argument("-z", type=int, required=True, help="The z argument")
    parser.add_argument("-e", type=int, required=True, help="The e argument")
    parser.add_argument("-b", type=int, required=True, help="The b argument")
    parser.add_argument("-s", type=str, required=True, help="The s argument")
    parser.add_argument("-p", type=str, required=True, help="The p argument")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.z, args.e, args.b, args.s, args.p)
