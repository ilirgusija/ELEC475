from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os

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

def single_point_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_folder = "data"
    dataset = KeypointDataset(root_folder, target_size=(256, 256))
    model = single_point()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_epochs = 30
    early_stopping_patience = 3
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    print(len(train_loader), len(test_loader), test_loader)
    
    output_dir="../single_point_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trained_model, train_losses, val_losses = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience, output_dir, device)
    plot_losses(train_losses, val_losses, "single_point", output_dir)
    
def custom_keypoint_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_folder = "data"
    dataset = HeatmapKeypointDataset(root_folder, target_size=(256, 256))
    model = CustomKeypointModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_epochs = 30
    early_stopping_patience = 3
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    print(len(train_loader), len(test_loader), test_loader)
    
    output_dir = "../heatmap_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trained_model, train_losses, val_losses = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience, output_dir, device)
    plot_losses(train_losses, val_losses, "CustomKeypoint", output_dir)
      
if __name__ == "__main__":

    # Call the main function with parsed arguments
    single_point_main()
    custom_keypoint_main()