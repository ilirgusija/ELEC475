from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
from PIL import Image

def plot_losses(train_losses, val_losses, model_name, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as a PNG file
    plot_path = os.path.join(output_dir, f"{model_name}_loss_plot.png")
    plt.savefig(plot_path)

    # Optionally, if you want to close the plot to free memory
    plt.close()

def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience, params_dir, device):
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    early_stopping_counter = 0
    print_interval = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % print_interval == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item()

        average_val_loss = running_val_loss / len(test_loader)
        val_losses.append(average_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
            
        if average_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {average_val_loss:.4f}. Saving model...")
            best_val_loss = average_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            checkpoint_path = os.path.join(params_dir, f'best_model_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1    

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs of no improvement in validation loss.")
            return model, train_losses, val_losses

    return model, train_losses, val_losses

def plot_image_with_keypoint(image_tensor, keypoint, target_size):
    plt.figure(figsize=(6, 6))

    # Convert tensor to numpy array and adjust dimensions
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

    plt.imshow(image_np)

    # Adjust keypoint coordinates for the plot
    # Assuming keypoint is already scaled according to the target_size
    keypoint_x, keypoint_y = keypoint[0] * target_size[0], keypoint[1] * target_size[1]

    # Create a red circle to mark the keypoint
    plt.scatter([keypoint_x], [keypoint_y], color='red', s=50)

    plt.title("Image with Keypoint")
    plt.show()

def check_images(root_dir):
    total_images = 0
    corrupt_images = 0

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                total_images += 1
                file_path = os.path.join(subdir, file)

                # Attempt to open the image
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify if it's a valid JPEG
                except (IOError, SyntaxError) as e:
                    corrupt_images += 1
                    print('Corrupt image:', file_path)

    print(f"Total images checked: {total_images}")
    print(f"Corrupt images found: {corrupt_images}")

def single_point_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    train_label_file = "train_noses.3.txt"
    test_label_file = "test_noses.txt"
    train_dataset = KeypointDataset(root_folder, train_label_file, target_size=(256, 256))
    test_dataset = KeypointDataset(root_folder, test_label_file, target_size=(256, 256))
    model = single_point()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    # idx = torch.randint(0, len(dataset), (1,)).item()
    # image, keypoint = dataset[idx]
    # plot_image_with_keypoint(image, keypoint, target_size=(256, 256))

    num_epochs = 30
    early_stopping_patience = 3
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # print(len(train_loader), len(test_loader), test_loader)
    
    output_dir="../single_point_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trained_model, train_losses, val_losses = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience, output_dir, device)
    plot_losses(train_losses, val_losses, "single_point", output_dir)
    
def plot_image_and_heatmap(image_tensor, heatmap_tensor):
    plt.figure(figsize=(12, 6))

    # Convert image tensor to numpy array and adjust dimensions (C, H, W) to (H, W, C)
    image = image_tensor.permute(1, 2, 0).numpy()

    # Convert heatmap tensor to numpy array and remove channel dimension (1, H, W) to (H, W)
    heatmap = heatmap_tensor.squeeze(0).numpy()

    # Plotting the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Resized Image")

    # Plotting the heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("Heatmap")

    plt.show()
    
def custom_keypoint_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_folder = "../data/oxford-iiit-pet-noses/images/"
    train_label_file = "train_noses.3.txt"
    test_label_file = "test_noses.txt"
    train_dataset = HeatmapKeypointDataset(root_folder, train_label_file, target_size=(256, 256))
    test_dataset = HeatmapKeypointDataset(root_folder, test_label_file, target_size=(256, 256))
    model = CustomKeypointModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    # idx = torch.randint(0, len(train_dataset), (1,)).item()
    # image, heatmap = train_dataset[idx]
    # plot_image_and_heatmap(image, heatmap)
    
    num_epochs = 30
    early_stopping_patience = 3
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    # print(len(train_loader), len(test_loader), test_loader)
    
    output_dir = "../heatmap_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trained_model, train_losses, val_losses = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience, output_dir, device)
    plot_losses(train_losses, val_losses, "CustomKeypoint", output_dir)
      
if __name__ == "__main__":

    # Call the main function with parsed arguments
    # single_point_main()
    custom_keypoint_main()