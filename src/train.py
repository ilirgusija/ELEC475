import argparse
import datetime
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from custom_dataset import custom_dataset
from AdaIN_net import encoder_decoder, AdaIN_net


def train(model, n_epochs, n_batches, optimizer, scheduler, ct_loader, st_loader, device):
    print("training...")
    
    model.train()
    model.to(device)
    
    c_losses_train=[]
    s_losses_train=[]
    losses_train=[]
    
    for epoch in range(1, n_epochs+1):
        loss_train = c_loss_train = s_loss_train = 0.0
        
        for b in range(1, n_batches+1):
            # print('epoch {}, batch {}'.format(epoch, b))
            content_images = next(iter(ct_loader)).to(device)
            style_images = next(iter(st_loader)).to(device)
            c_loss, s_loss = model(content_images, style_images)
            
            loss = c_loss + s_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            loss_train += loss.item()
            c_loss_train += c_loss.item()
            s_loss_train += s_loss.item()
            
            optimizer.step()
            
            scheduler.step()
            
            c_losses_train += [loss_train/len(ct_loader)]
            s_losses_train += [loss_train/len(st_loader)]
            losses_train += [loss_train/len(st_loader)]
        
            print('{} Epoch {}, Batch {}, Content loss {}, Style loss {}'.format(datetime.datetime.now(), epoch, b, c_loss_train/len(ct_loader), s_loss_train/len(st_loader)))
    return losses_train, c_losses_train, s_losses_train  
        

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def main(content_dir, style_dir, gamma, n_epochs, batch_size, load_encoder, save_decoder, plot_decoder, device):
    model = AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)
    
    # Summarize model and data
    summary(model, (1, 256*256))
    
    content_tf = train_transform()
    style_tf = train_transform()
    
    content_dataset = custom_dataset(content_dir, transform=content_tf)
    style_dataset = custom_dataset(style_dir, transform=style_tf)
    
    content_loader = DataLoader(content_dataset, batch_size, shuffle=True)
    style_loader = DataLoader(style_dataset, batch_size, shuffle=True)
 
    optimizer = Adam(model.decoder.parameters(), lr=0.01)
    sched = lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)
    
    decoder = encoder_decoder.decoder
    encoder = encoder_decoder.encoder

    encoder.load_state_dict(torch.load(load_encoder))
    my_model = AdaIN_net(encoder, decoder)
    
    loss, c_loss, s_loss  = train(model, n_epochs, batch_size, optimizer, sched, content_loader, style_loader, device)
    
    torch.save(my_model.decoder.state_dict(), save_decoder)
    
    # Plot loss curve
    plt.plot(loss, label='Total Loss', marker='o', linestyle='-', color='red')
    plt.plot(c_loss, label='Content Loss', marker='o', linestyle='-', color='blue')
    plt.plot(s_loss, label='Style Loss', marker='o', linestyle='-', color='yellow')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.savefig(plot_decoder)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for style transfer")
    parser.add_argument('-content_dir', type=str, required=True, help='Path to the content dataset directory')
    parser.add_argument('-style_dir', type=str, required=True, help='Path to the style dataset directory')
    parser.add_argument('-gamma', type=float, default=1.0, help='Gamma value for style transfer')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('-l', '--load_encoder', type=str, default="encoder.pth", help='Path to load the encoder model')
    parser.add_argument('-s', '--save_decoder', type=str, default="decoder.pth", help='Path to save the decoder model')
    parser.add_argument('-p', '--plot_decoder', type=str, default="decoder.png", help='Path to save the decoder image')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'
    main(args.content_dir, args.style_dir, args.gamma, args.epochs, args.batch_size, args.load_encoder, args.save_decoder, args.plot_decoder, device)
