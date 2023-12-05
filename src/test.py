import torch
import matplotlib.pyplot as plt
from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary
import torch
import numpy as np
from scipy.spatial.distance import euclidean



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
    root_folder = "../data/oxford-iiit-pet-noses/images"
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
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    dataset = KeypointDataset(root_folder, "test_noses.txt", target_size=(256, 256))
    num_samples = len(dataset)
    test_model = single_point()
    checkpoint = torch.load('../single_point_output/best_model_checkpoint.pth')
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    test_model.eval()

    idx = torch.randint(0, num_samples, (1,)).item()
    image, keypoints = dataset[idx]
    image_to_model = image.unsqueeze(0)

    with torch.no_grad():
        output_coords = test_model(image_to_model)

    coords_pred = output_coords.squeeze().numpy()  #

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(image))
    plt.title('Original Image')

    # Scatter plot for ground truth and predicted points
    plt.scatter(keypoints[0], keypoints[1], color='red', marker='x')
    plt.scatter(coords_pred[0], coords_pred[1], color='blue', marker='x')
    plt.legend()
    plt.title('Ground Truth and Predicted Points')
    
if __name__ == "__main__":
    # test_single_main()
    test_heatmap_main()