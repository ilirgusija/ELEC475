import torch
import matplotlib.pyplot as plt
from KeypointDataset import  HeatmapKeypointDataset
from model import CustomKeypointModel
import warnings
import torch.cuda
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch.nn as nn
import os
import time
warnings.filterwarnings("ignore")

def plot_image_and_heatmap(image_tensor, heatmap_tensor, target_heatmap_tensor, true_keypoint, predicted_keypoint, output_dir, i):
    plt.figure(figsize=(18, 6))

    # Detach tensors from the computation graph and convert to numpy arrays
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    heatmap = heatmap_tensor.detach().cpu().squeeze(0).numpy()
    target_heatmap = target_heatmap_tensor.detach().cpu().squeeze(0).numpy()

    # Unpack keypoints
    true_y, true_x = true_keypoint
    pred_y, pred_x = predicted_keypoint

    # Plotting the image with keypoints
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.scatter([true_x, pred_x], [true_y, pred_y], c=['blue', 'red'])  # True keypoint in blue, predicted in red
    plt.title("Resized Image with Keypoints")

    # Plotting the predicted heatmap with keypoint
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.scatter(pred_x, pred_y, c='red')  # Predicted keypoint
    plt.title("Predicted Heatmap with Keypoint")

    # Plotting the target heatmap with keypoint
    plt.subplot(1, 3, 3)
    plt.imshow(target_heatmap, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.scatter(true_x, true_y, c='blue')  # True keypoint
    plt.title("Target Heatmap with Keypoint")

    # Save the plot as a PNG file
    plot_path = os.path.join(output_dir, f"sample_{i}_results.png")
    plt.savefig(plot_path)

    # Optionally, if you want to close the plot to free memory
    plt.close()

def compute_batch_centroids(heatmaps):
    # print(f"validating heatmap shape: {heatmaps.shape}")
    batch_size, _, height, width = heatmaps.shape
    device = heatmaps.device
    y_indices, x_indices = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))

    # print(f"y shape {y_indices.shape}")

    # Reshape to allow broadcasting
    y_indices = y_indices.expand(batch_size, height, width)
    x_indices = x_indices.expand(batch_size, height, width)

    total_heat = heatmaps.sum(dim=[2, 3], keepdim=True)
    heatmaps = heatmaps.squeeze(1)
    y_center = (y_indices * heatmaps).sum(dim=[1, 2]) / total_heat.squeeze()
    x_center = (x_indices * heatmaps).sum(dim=[1, 2]) / total_heat.squeeze()

    return y_center, x_center

def test(test_loader, model, loss_fn, device):
    all_distances = []
    all_samples = []
    true_keypoints = []
    pred_keypoints = []
    total_loss = 0.0
    total_inference_time = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.perf_counter() 
            outputs = model(inputs).to(device)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            end_time = time.perf_counter() 
            
            # Compute centroids for the whole batch
            pred_y_centers, pred_x_centers = compute_batch_centroids(outputs)
            true_y_centers, true_x_centers = compute_batch_centroids(labels)

            # Calculate distances for the batch
            distances = torch.sqrt((true_x_centers - pred_x_centers)**2 + (true_y_centers - pred_y_centers)**2)
            all_distances.extend(distances.cpu().numpy())

            # Collect samples for visualization
            for i in range(inputs.shape[0]):
                all_samples.append((inputs[i], labels[i], outputs[i]))
                true_keypoints.append((true_y_centers[i].item(), true_x_centers[i].item())) 
                pred_keypoints.append((pred_y_centers[i].item(), pred_x_centers[i].item())) 

    mean_loss = total_loss / len(test_loader)
    mean_distance = np.mean(all_distances)
    std_distance = np.std(all_distances)
    min_distance = np.min(all_distances)
    max_distance = np.max(all_distances)

    return mean_loss, total_inference_time, min_distance, max_distance, mean_distance, std_distance, all_samples, true_keypoints, pred_keypoints

def main(root_folder, model_pth_path, labels_file, output_dir, b, device):
   
    dataset = HeatmapKeypointDataset(root_folder, labels_file, target_size=(256, 256))
    test_loader = DataLoader(dataset, batch_size=b, shuffle=False)
    
    test_model = CustomKeypointModel()
    checkpoint = torch.load(model_pth_path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device).eval()
    
    mean_loss, time, min_distance, max_distance, mean_distance, std_distance, pred_heatmaps, true_keypoints, pred_keypoints  = test(test_loader, test_model, nn.MSELoss(), device)
    
    # Print test results
    print(f"Inference time {time:.15f}, Mean Loss: {mean_loss}, Min Distance: {min_distance}, Max Distance: {max_distance}, Mean Distance: {mean_distance}, Std Distance: {std_distance}")

    num_samples_to_visualize = 3
    for i in range(num_samples_to_visualize):
        image_tensor, target_heatmap_tensor, predicted_heatmap_tensor = pred_heatmaps[i]
        pred_k = pred_keypoints[i]
        true_k = true_keypoints[i]
        plot_image_and_heatmap(image_tensor.cpu(), predicted_heatmap_tensor.cpu(), target_heatmap_tensor.cpu(), true_k, pred_k, output_dir, i)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Keypoint Model")
    parser.add_argument('--root_folder', type=str, required=True, help="Path to the root folder containing images")
    parser.add_argument('--model_pth_path', type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the model params save location")
    parser.add_argument('--labels_file', type=str, required=True, help="Path to the labels file")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for inference")

    args = parser.parse_args()
    labels_file = args.labels_file
    root_folder = args.root_folder
    model_pth_path = args.model_pth_path
    output_dir = args.output_dir
    b = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(root_folder, model_pth_path, labels_file, output_dir, b, device)