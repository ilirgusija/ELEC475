import argparse
import sys
import datetime
from classifier import object_classifier, encoder_decoder
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

def pad_roi(roi, target_size=(150, 150)):
    height, width = roi.shape[:2]
    pad_height = max(target_size[0] - height, 0)
    pad_width = max(target_size[1] - width, 0)

    # Pad the ROI on the right and bottom to reach the target size
    padded_roi = F.pad(roi, pad=(0, pad_width, 0, pad_height), mode='constant', value=0)
    return padded_roi

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
            model = object_classifier(encoder, encoder_decoder.decoder10van)
        elif(dataset==100):
            model = object_classifier(encoder, encoder_decoder.decoder100van)
        params = model.decoder.parameters()
    elif(modelType=='modified'):
        model = object_classifier(encoder=encoder_decoder.encoder_resnet, num_classes=dataset)
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