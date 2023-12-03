from KeypointDataset import KeypointDataset, HeatmapKeypointDataset
from model import CustomKeypointModel, single_point
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Save the model if validation loss improves
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
            torch.save(checkpoint, '../params/best_model_checkpoint.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs of no improvement in validation loss.")
            return model, train_losses, val_losses
            break

    return model, train_losses, val_losses


def main():
    
    root_folder = "data"
    dataset = HeatmapKeypointDataset(root_folder, target_size=(256, 256))
    num_samples = len(dataset)
    model = CustomKeypointModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    input_image = torch.randn(1, 3, 256, 256)
    input_image = input_image.to(device)

    with torch.no_grad():
        output_heatmap = model(input_image)

    print("Output shape:", output_heatmap.shape)
    summary(model, (3, 256, 256))
    
    dataset = KeypointDataset(root_folder, target_size=(256, 256))
    num_samples = len(dataset)

    for i in range(3):
        index = torch.randint(0, len(dataset), (1,)).item()
        image, keypoints = dataset[index]

        keypoints = keypoints * torch.tensor([image.shape[2], image.shape[1]])

        # Visualize the image with keypoints
        plt.imshow(F.to_pil_image(image))
        plt.scatter(keypoints[0], keypoints[1], c='red', marker='o')
        plt.title(f"Sample {index + 1}")
        plt.axis('off')
        plt.show()
        
    model = single_point()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_epochs = 30
    early_stopping_patience = 3
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(len(train_loader), len(test_loader), test_loader)
    
    trained_model, train_losses, val_losses = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, early_stopping_patience)
    plot_losses(train_losses, val_losses)
    
    
    
if __name__ == "__main__":

    # Call the main function with parsed arguments
    main()