import torch
import matplotlib.pyplot as plt
from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import matplotlib.pyplot as plt
import warnings
import torch.cuda
from torch.utils.data import DataLoader
import numpy as np
import numpy as np
from scipy.spatial.distance import euclidean
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


def compute_heatmap_centroid(heatmap):
    # Generate a grid of coordinates (x, y)
    y_indices, x_indices = np.indices(heatmap.shape)

    # Compute the weighted average of the coordinates, using heatmap values as weights
    total_heat = heatmap.sum()
    x_center = (x_indices * heatmap).sum() / total_heat
    y_center = (y_indices * heatmap).sum() / total_heat

    return int(y_center), int(x_center)

def evaluate_localization_accuracy(model, dataset, device):
    model.to(device)
    model.eval()

    distances = []

    for image, target_heatmap in dataset:
        # Predict the heatmap
        image = image.to(device)

        with torch.no_grad():
            output_heatmap = model(image).cpu().squeeze().numpy()

        # Get the ground truth keypoint location (assuming it's the maximum point)
        true_y, true_x = np.unravel_index(np.argmax(target_heatmap.squeeze().numpy()), target_heatmap.shape[1:])

        # Get the predicted centroid of the heatmap
        pred_y, pred_x = compute_heatmap_centroid(output_heatmap)

        # Calculate the Euclidean distance and store it
        distance = euclidean((true_x, true_y), (pred_x, pred_y))
        distances.append(distance)

    # Calculate statistics
    min_distance = torch.min(distances)
    mean_distance = torch.mean(distances)
    max_distance = torch.max(distances)
    std_distance = torch.std(distances)

    return min_distance, mean_distance, max_distance, std_distance

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
    
if __name__ == "__main__":
    test_heatmap_main()