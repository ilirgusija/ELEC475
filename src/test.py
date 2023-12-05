import torch
import matplotlib.pyplot as plt
from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms as transforms
import torch.cuda
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
warnings.filterwarnings("ignore")


def calculate_distances(predictions, labels):
    distances = [torch.norm(pred - label).unsqueeze(0) for pred, label in zip(predictions, labels)]
    distances_tensor = torch.cat(distances)
    return distances_tensor

def test(test_loader, model, loss_fn, device):
    print("testing...")
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs)  # Assuming outputs are in numpy format
            all_labels.append(labels)  # Assuming labels are in numpy format

    mean_loss = total_loss / len(test_loader)

    # Convert lists to tensors
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Calculate distances
    distances = calculate_distances(all_predictions_tensor, all_labels_tensor)

    min_distance = torch.min(distances)
    max_distance = torch.max(distances)
    mean_distance = torch.mean(distances)
    std_distance = torch.std(distances)

    return mean_loss, min_distance, max_distance, mean_distance, std_distance, all_predictions

def test_heatmap_main():
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    dataset = HeatmapKeypointDataset(root_folder, "test_noses.txt", target_size=(256, 256))
    test_model = CustomKeypointModel()
    checkpoint = torch.load('../heatmap_output/best_model_checkpoint.pth')
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()

    idx = torch.randint(0, len(dataset), (1,)).item()
    image, heatmap = dataset[idx]
    image_to_model = image.unsqueeze(0)

    with torch.no_grad():
        output_heatmap = test_model(image_to_model)

    heatmap_np = output_heatmap.numpy()
    heatmap_np = heatmap_np.squeeze()

    print('here', heatmap_np)
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Original Image')

    # Plot the generated heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
    plt.title('Generated Heatmap')

    plt.show()
        
def test_single_main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
        ]
    )

    batch_size = 512
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    dataset = KeypointDataset(root_folder, "test_noses.txt", target_size=(256, 256))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('../single_point_output/best_model_checkpoint.pth', map_location=torch.device(device))
    test_model = single_point()
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model = test_model.to(device)
    loss_fn = nn.MSELoss()
    test_model.eval()
    mean_loss, min_distance, max_distance, mean_distance, std_distance, preds = test(test_loader, test_model, loss_fn, device)

    print("Mean Loss:", mean_loss)
    print("Min Euclidean Distance:", min_distance)
    print("Max Euclidean Distance:", max_distance)
    print("Mean Euclidean Distance:", mean_distance)
    print("Standard Deviation of Euclidean Distances:", std_distance)
    
    plt.title('Ground Truth and Predicted Points')
    
    plt.show()
    
if __name__ == "__main__":
    test_single_main()
    # test_heatmap_main()