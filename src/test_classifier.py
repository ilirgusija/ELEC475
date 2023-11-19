import argparse
from classifier import object_classifier, backends
import torch
from BinaryROIsDataset import BinaryROIsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
    
def main(classifier_file, data_dir, batch_size, model, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transform.Resize((150,150)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Create dataset instances with padding option
    test_dataset = BinaryROIsDataset(data_dir=data_dir, data_type='test', transform=transform, padding=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = object_classifier(encoder=backends.encoder_se_resnet)
    model.load_state_dict(torch.load(classifier_file))
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        total = 0
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            
            # Update correct counts
            correct += (predicted == labels).sum().item()
            
    # Calculate the errors
    print("Correct: {}, Total: {}".format(correct, total))
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
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    parser.add_argument('-b', '--batch_size', type=int, default=20, help='Batch size for testing')
    parser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')
    # Parse the arguments
    args = parser.parse_args()
    if(args.cuda=='Y'):
        device='cuda'
    else:
        device='cpu'
    main(args.classifier_params, args.data_dir, args.batch_size, device)