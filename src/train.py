import argparse
import datetime
from AdaIN_net import encoder_decoder, AdaIN_net

def train(model, n_epochs, optimizer, scheduler, loss_fn, train_loader, device):
    model.train()
    model.to(device)
    losses_train=[]
    
    for epoch in range(1, n_epochs+1):
        print('epoch', epoch)
        loss_train=0.0
        for inputs, true_out in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, true_out)
            
            optimizer.zero_grad()
            loss.backward()
            loss_train+=loss.item()
            
            optimizer.step()
            
        scheduler.step()
        
        losses_train += [loss_train/len(train_loader)]
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    return losses_train    
        

def main(content_dir, style_dir, gamma, n_epochs, batch_size, load_encoder, save_decoder, plot_decoder, device):
    model = AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)
    
    train(model, n_epochs, device)


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
    main(args.content_dir, args.style_dir, args.gamma, args.e, args.b, args.l, args.s, args.p, device)
