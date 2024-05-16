import datetime
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model_v import encoder_decoder, image_classifier
from model_m import encoder_decoder as mod_encoder
from model_m import mod_NN
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import sys


def import_dataset(batch_size=16, tenOrHundred=10):
    # Load train set
    # Data transform (you can add more transforms if you wish)
    transform = transforms.Compose([transforms.ToTensor()])
    
    if (tenOrHundred==10):
        print("loading cifar10")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Confirm dataset size
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of test samples: {len(testset)}")
    
    return trainloader, testloader

def train(model, n_epochs, loss_fn, optimizer, scheduler, train_loader, device):
    # Iterating through batches
    print('training ...')
    model.to(device)
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, desired in train_loader:
            imgs = imgs.to(device) 
            desired = desired.to(device) 
            
            # compute output
            outputs = model(imgs)
            loss = loss_fn(outputs, desired)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.item()
            
            optimizer.step()

        scheduler.step()

        losses_train += [loss_train/len(train_loader)] # average out loss over the epoch
        

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    return losses_train

def main(gamma, n_epochs, dataset, batch_size, save_encoder, load_encoder, save_decoder, plot_model, modelType, device):
    train_loader, _ = import_dataset(batch_size, dataset) 
    
    
    if(modelType=='vanilla'):
        encoder = encoder_decoder.encoder
        encoder.load_state_dict(torch.load(load_encoder))
        if(dataset==10):
            model = image_classifier(encoder, encoder_decoder.decoder10van)
        elif(dataset==100):
            model = image_classifier(encoder, encoder_decoder.decoder100van)
        params = model.decoder.parameters()
    elif(modelType=='modified'):
        model = mod_NN(encoder=mod_encoder.encoder, num_classes=dataset)
        params = model.parameters()
    else:
        print("No valid model type chosen, aborting...")
        sys.exit()
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=0.001,  weight_decay=1e-5)
    sched = StepLR(optimizer, step_size=10, gamma=gamma)
    
    loss = train(model, n_epochs, loss_fn, optimizer, sched, train_loader, device)
    
    if(modelType == 'modified' and save_encoder):
        torch.save(model.encoder.state_dict(), save_encoder)
        
    torch.save(model.decoder.state_dict(), save_decoder)
    
    # Plot loss curve
    plt.clf()
    plt.figure(figsize=(12, 7))
    plt.plot(loss, label='Total Loss', color='red')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.savefig(plot_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Image classification")
    parser.add_argument('-gamma', type=float, default=1.0, help='Gamma value for scheduler')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('-d', '--dataset', type=int, default=10, help='Which CIFAR dataset, 10 or 100? (defaults to 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('-se', '--save_encoder', type=str, default=None, help='Path to save the backend model')
    parser.add_argument('-l', '--load_encoder', type=str, default="encoder.pth", help='Path to load the backend model')
    parser.add_argument('-sd', '--save_decoder', type=str, default="decoder.pth", help='Path to save the frontend model')
    parser.add_argument('-p', '--plot_model', type=str, default="decoder.png", help='Path to save the frontend image')
    parser.add_argument('-m', '--model_type', type=str, default='vanilla', help='Which model type?')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'
    main(args.gamma, args.epochs, args.dataset, args.batch_size, args.save_encoder, args.load_encoder, args.save_decoder, args.plot_model, args.model_type, device)