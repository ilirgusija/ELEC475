import torch
import matplotlib.pyplot as plt
from model import CustomKeypointModel
from KeypointDataset import KeypointDataset
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def main():
    root_folder = "data"
    dataset = KeypointDataset(root_folder, target_size=(256, 256))
    num_samples = len(dataset)
    test_model = CustomKeypointModel()
    test_model.load_state_dict(torch.load('../params/best_model.pth'))
    test_model.eval()

    idx = torch.randint(0, num_samples, (1,)).item()
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
 ###################################################################################   
    model.eval()

    idx = torch.randint(0, len(dataset), (1,)).item()
    image, keypoints = dataset[idx]
    image_to_model = image.unsqueeze(0)

    with torch.no_grad():
        output_heatmap = test_model(image_to_model)

    heatmap_np = output_heatmap.squeeze().numpy()
    print('here', heatmap_np)

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(image))
    plt.title('Original Image')

    # Plot the generated heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
    plt.title('Generated Heatmap')

    plt.show()
    
 ###################################################################################   
    model = single_point()
    model.eval()

    idx = torch.randint(0, len(dataset), (1,)).item()
    image, keypoints = dataset[idx]
    image_to_model = image.unsqueeze(0)

    with torch.no_grad():
        output_coords = model(image_to_model)

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

    plt.show()
