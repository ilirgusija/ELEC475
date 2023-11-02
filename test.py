import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch
import torchvision.transforms as transforms
from model import image_classifier, encoder_decoder
from train import import_dataset


if __name__ == '__main__':

    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_file', type=str, help='encoder weight file')
    parser.add_argument('-decoder_file', type=str, help='decoder weight file')
    parser.add_argument('-d', '--dataset', type=int, default=10,
                        help='Which CIFAR dataset, 10 or 100? (defaults to 10)')
    parser.add_argument('-cuda', type=str, help='[y/N]')

    opt = parser.parse_args()
    encoder_file = opt.encoder_file
    decoder_file = opt.decoder_file
    dataset = opt.dataset
    _, test_loader = import_dataset(tenOrHundred=dataset)
    use_cuda = False
    if opt.cuda == 'y' or opt.cuda == 'Y':
        use_cuda = True
    out_dir = './output/'
    os.makedirs(out_dir, exist_ok=True)

    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
    if(dataset==10):
        decoder = encoder_decoder.decoder10van
    elif(dataset==100):
        decoder = encoder_decoder.decoder100van

    if torch.cuda.is_available() and use_cuda:
        print('using cuda ...')
        decide='cuda'
    else:
        print('using cpu ...')
    model = image_classifier(encoder, decoder)
    model.to(device=device)
    model.eval()

    print('model loaded OK!')


    with torch.no_grad():
        total = 0
        correct_top1 = 0
        correct_top5 = 0
        
        for images, labels in test_loader:
            print(f"Initial shape of images: {images.shape}, labels: {labels.shape}")  # Shape of images and labels
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            print(f"Shape of outputs: {outputs.shape}")  # Shape of outputs
            print(f"Sample outputs: {outputs[0]}")  # Sample raw scores (logits) for the first image in the batch
            
            # For Top-1: Get the index of the highest output score
            _, predicted_top1 = torch.max(outputs, 1)
            
            print(f"Predicted Top-1 labels: {predicted_top1}")  # Top-1 predicted labels
            
            # For Top-5: Sort the predictions and get top 5 indices
            _, sorted_indices = torch.sort(outputs, descending=True)
            top5_indices = sorted_indices[:, :5]
            
            print(f"Top-5 predicted labels for first image: {top5_indices[0]}")  # Top-5 predicted labels for the first image
            
            # Update total count
            total += labels.size(0)
            
            # Update correct counts
            correct_top1 += (predicted_top1 == labels).sum().item()
            
            for i in range(labels.size(0)):
                if labels[i] in top5_indices[i]:
                    correct_top5 += 1

            print(f"True labels: {labels}")  # True labels
            print(f"Correct Top-1 so far: {correct_top1}, Correct Top-5 so far: {correct_top5}")  # Correct counts
            
            
    # Calculate the errors
    print("Correct top 1: {}, Correct top 5: {}, Total: {}".format(correct_top1,correct_top5,total))
    top1_error = 1.0 - (correct_top1 / float(total))
    top5_error = 1.0 - (correct_top5 / float(total))

    print(f"Final Top-1 error: {top1_error:.3f}, Top-5 error: {top5_error:.3f}")

