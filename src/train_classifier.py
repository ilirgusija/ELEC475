import argparse
import datetime
from classifier import object_classifier, backends
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from BinaryROIsDataset import BinaryROIsDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

def evaluate(model, test_loader, device, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for imgs, desired in test_loader:
            imgs = imgs.to(device)
            desired = desired.to(device).float()
            outputs = model(imgs).squeeze(dim=1)
            loss = loss_fn(outputs, desired)
            total_loss += loss.item()
            total += 1
    return total_loss / total  # Return average loss

def train(model, n_epochs, loss_fn, optimizer, scheduler, train_loader, test_loader, device):
    # Iterating through batches
    print('training ...')
    model.to(device)
    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        model.train()
        for i, (imgs, desired) in enumerate(train_loader):
            imgs = imgs.to(device)
            desired = desired.to(device).float() 
            
            # compute output
            outputs = model(imgs)
            outputs = outputs.squeeze(dim=1)
                
            loss = loss_fn(outputs, desired)
            
            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.item()
            
            optimizer.step()
            # if i==100:
            #     sys.exit()
            print('{} Epoch {}, Batch{}, Training loss {}'.format(datetime.datetime.now(), epoch, i, loss.item()))

        scheduler.step()
        avg_train_loss = loss_train / len(train_loader)
        losses_train.append(avg_train_loss)

        # Evaluate on test set for validation loss
        val_loss = evaluate(model, test_loader, device, loss_fn)
        
        losses_val.append(val_loss)

        print(f'{datetime.datetime.now()} Epoch {epoch}, Training loss: {avg_train_loss}, Validation loss: {val_loss}')

    return losses_train, losses_val
    

def main(gamma, n_epochs, model_type, learning_rate, data_dir, batch_size, save_model, plot_model, sampling, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((150, 150), antialiasing=True),
    ])    

    # if sampling=="downsample":
    #     train_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='train', transform=transform, balance_data=True, target_balance_ratio=0.5)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # else:
    train_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='train', transform=transform, balance_data=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    # Test DataLoader
    test_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    if model_type=='resnet_18':
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))
    else:
        model = object_classifier(encoder=getattr(backends, model_type, None))
        model = object_classifier(encoder=getattr(backends, encoder, None))
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    sched = StepLR(optimizer, step_size=10, gamma=gamma)
    
    losses_train, losses_val = train(model, n_epochs, loss_fn, optimizer, sched, train_loader, test_loader, device)

    torch.save(model.state_dict(), save_model)
    
    # Plot loss curves
    plt.figure(figsize=(12, 7))
    plt.plot(losses_train, label='Training Loss', color='red')
    plt.plot(losses_val, label='Validation Loss', color='blue')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(plot_model)

if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the parent directory (assuming the script is in 'src')
    parent_dir = os.path.dirname(current_script_dir)

    # Set the default data directory
    default_data_dir = os.path.join(parent_dir, "data/Kitti8ROIs")
    parser = argparse.ArgumentParser(description="Training script for Image classification")
    parser.add_argument('-gamma', type=float, default=1.0, help='Gamma value for scheduler')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('-m', '--model_type', choices=['resnet', 'se_resnet', 'se_resneXt', 'resnet_18'], default='resnet', help='Classifier type')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('-s', '--save_model', type=str, default="model.pth", help='Path to save the model')
    parser.add_argument('-p', '--plot_model', type=str, default="model_loss.png", help='Path to save the loss plot')
    parser.add_argument('-d_p', choices=['downsample', 'neither'], default='', help='Whether to downsample majority or leave the dataset alone (downsample/neither)')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'

    main(args.gamma, args.epochs, args.model_type, args.learning_rate, args.data_dir, args.batch_size, args.save_model, args.plot_model, args.d_p, device)