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
import sys
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

def train(model, n_epochs, loss_fn, optimizer, scheduler, train_loader, device):
    # Iterating through batches
    print('training ...')
    model.to(device)
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for i, (imgs, desired) in enumerate(train_loader):
            imgs = imgs.to(device)
            desired = desired.to(device).float() 
            
            # compute output
            outputs = model(imgs)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, desired)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.item()
            
            optimizer.step()
            print('{} Epoch {}, Batch{}, Training loss {}'.format(datetime.datetime.now(), epoch, i, loss_train/len(train_loader)))

        scheduler.step()

        losses_train += [loss_train/len(train_loader)] # average out loss over the epoch
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    return losses_train
    
    
def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for item in labels:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  

def main(gamma, n_epochs, model_type, learning_rate, data_dir, batch_size, save_model, plot_model, sampling, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150,150), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    if sampling is not True:
        train_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='train', transform=transform, balance_data=True, target_balance_ratio=0.5)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        # Calculate weights for each sample
        train_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='train', transform=transform, balance_data=False)
        class_counts = train_dataset._count_labels()[0]
        print("Label Counts: ", class_counts)
        class_weights = {class_id: 1.0/count for class_id, count in class_counts.items()}
        weights = [class_weights[label] for _, label in train_dataset.labels]

        # Create a WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)

    model = object_classifier(encoder=getattr(backends, model_type, None))
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)
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
    parser.add_argument('-m', '--model_type', choices=['resnet', 'se_resnet', 'se_resneXt'], default='resnet', help='Classifier type')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('-s', '--save_model', type=str, default="model.pth", help='Path to save the model')
    parser.add_argument('-p', '--plot_model', type=str, default="model_loss.png", help='Path to save the loss plot')
    parser.add_argument('-d_p', choices=['downsample', 'weighted'], default='weighted', help='Whether to downsample majority or use weighted sampler (downsample/weighted)')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'
        
    if args.d_p=='weighted':
        sampling=True
    else:
        sampling=False
    main(args.gamma, args.epochs, args.model_type, args.learning_rate, args.data_dir, args.batch_size, args.save_model, args.plot_model, sampling, device)