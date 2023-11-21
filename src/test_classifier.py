import argparse
from classifier import object_classifier, backends
import torch
from BinaryROIsDataset import BinaryROIsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch.nn as nn

    
def main(classifier_file, data_dir, batch_size, model_type, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150,150))
    ])

    # Create dataset instances with padding option
    test_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = object_classifier(encoder=getattr(backends, model_type, None))
    model.load_state_dict(torch.load(classifier_file))
    
    loss_fn = nn.BCELoss()
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        total = 0
        loss_test = 0.0
        correct = 0 
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images)
            outputs = outputs.squeeze()
            
            loss = loss_fn(outputs, labels)
            
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            loss_test+=loss.item()
            # Update correct counts
            correct += (predicted == labels).sum().item()
            
    # Calculate the errors
    print("Correct: {}, Total: {}".format(correct, total))
    print("Test Loss: {}".format(loss_test/len(test_loader)))
    
    error = 1.0 - (correct / float(total))

    print(f"Final error: {error:.3f}")

if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the parent directory (assuming the script is in 'src')
    parent_dir = os.path.dirname(current_script_dir)

    # Set the default data directory
    default_classifier = os.path.join(parent_dir, "output/classifier.pth")
    default_data_dir = os.path.join(parent_dir, "data/Kitti8ROIs")
    parser = argparse.ArgumentParser(description="Training script for Image classification")
    parser.add_argument('-c', '--classifier_params', type=str, default=default_classifier, help='Classifier weights')
    parser.add_argument('-m', '--model_type', choices=['resnet', 'se_resnet', 'se_resneXt'], default='resnet', help='Classifier type')
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for testing')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    device='cpu'
    if(args.cuda=='Y'):
        device='cuda'
    main(args.classifier_params, args.data_dir, args.batch_size, args.model_type, device)