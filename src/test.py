import torch
import matplotlib.pyplot as plt
from KeypointDataset import  HeatmapKeypointDataset
from model import CustomKeypointModel
import warnings
import torch.cuda
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import euclidean

warnings.filterwarnings("ignore")

def compute_batch_centroids(heatmaps):
    print(f"validating heatmap shape: {heatmaps.shape}")
    _, _, height, width = heatmaps.shape
    y_indices, x_indices = torch.meshgrid(torch.arange(height), torch.arange(width))

    # Reshape to allow broadcasting
    y_indices = y_indices.view(1, height, width)
    x_indices = x_indices.view(1, height, width)

    total_heat = heatmaps.sum(dim=[1, 2], keepdim=True)
    y_center = (y_indices * heatmaps).sum(dim=[1, 2]) / total_heat.squeeze()
    x_center = (x_indices * heatmaps).sum(dim=[1, 2]) / total_heat.squeeze()

    return y_center, x_center

def test(test_loader, model, loss_fn, device):
    all_distances = []
    all_samples = []
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Compute centroids for the whole batch
            pred_y_centers, pred_x_centers = compute_batch_centroids(outputs)
            true_y_centers, true_x_centers = compute_batch_centroids(labels)

            # Calculate distances for the batch
            distances = torch.sqrt((true_x_centers - pred_x_centers)**2 + (true_y_centers - pred_y_centers)**2)
            all_distances.extend(distances.cpu().numpy())

            # Collect samples for visualization
            for i in range(inputs.shape[0]):
                all_samples.append((inputs[i], labels[i], outputs[i]))


    mean_loss = total_loss / len(test_loader)
    mean_distance = torch.mean(all_distances)
    std_distance = torch.std(all_distances)
    min_distance = np.min(all_distances)
    max_distance = np.max(all_distances)

    return mean_loss, min_distance, max_distance, mean_distance, std_distance, all_samples

def main(root_folder, model_pth_path, labels_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = HeatmapKeypointDataset(root_folder, labels_file, target_size=(256, 256))
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    test_model = CustomKeypointModel()
    checkpoint = torch.load(model_pth_path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device).eval()
    
    mean_loss, min_distance, max_distance, mean_distance, std_distance, all_predictions = test(test_loader, test_model, nn.MSELoss(), device)
    
    # Print test results
    print(f"Mean Loss: {mean_loss}, Min Distance: {min_distance}, Max Distance: {max_distance}, Mean Distance: {mean_distance}, Std Distance: {std_distance}")

    idx = torch.randint(0, len(dataset), (1,)).item()
    image, heatmap = dataset[idx]
    image_to_model = image.unsqueeze(0)

    with torch.no_grad():
        output_heatmap = test_model(image_to_model)

    heatmap_np = output_heatmap.numpy()
    heatmap_np = heatmap_np.squeeze()
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Original Image')

    # Plot the generated heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
    plt.title('Generated Heatmap')

    plt.show()
    
if __name__ == "__main__":
    labels_file = "test_noses.txt"
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    model_pth_path = '../heatmap_output/best_model_checkpoint.pth'
    test_heatmap_main()