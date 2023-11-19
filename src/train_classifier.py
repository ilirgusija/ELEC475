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
import os

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
    
def main(gamma, n_epochs, data_dir, batch_size, save_model, plot_model, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150,150)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Create dataset instances with padding option
    train_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='train', transform=transform, padding=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = object_classifier(encoder=backends.encoder_se_resnet)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-5)
    sched = StepLR(optimizer, step_size=10, gamma=gamma)
    
    loss = train(model, n_epochs, loss_fn, optimizer, sched, train_loader, device)

    torch.save(model.state_dict(), save_model)
    
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
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the parent directory (assuming the script is in 'src')
    parent_dir = os.path.dirname(current_script_dir)

    # Set the default data directory
    default_data_dir = os.path.join(parent_dir, "data/Kitti8ROIs")
    parser = argparse.ArgumentParser(description="Training script for Image classification")
    parser.add_argument('-gamma', type=float, default=1.0, help='Gamma value for scheduler')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('-s', '--save_model', type=str, default="model.pth", help='Path to save the model')
    parser.add_argument('-p', '--plot_model', type=str, default="model_loss.png", help='Path to save the loss plot')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'
    main(args.gamma, args.epochs, args.data_dir, args.batch_size, args.save_model, args.plot_model, device)