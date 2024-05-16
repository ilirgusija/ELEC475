import argparse
import torch
import matplotlib.pyplot as plt
from model import MLP
from torchvision import transforms, datasets

def run_test(model, test_set):
    img, _ = test_set[0]
    
    img_batch = img.unsqueeze(0).view(1, -1)
    
    # Run the model
    with torch.no_grad():
        output_batch = model(img_batch)
    
    # Unflatten the input and output to visualize it as an image
    img = img.squeeze().numpy()
    output_img = output_batch.view(1, 28, 28).squeeze().numpy()
    
    f = plt.figure() 
    f.add_subplot(1,2,1) 
    plt.imshow(img, cmap='gray') 
    f.add_subplot(1,2,2) 
    plt.imshow(output_img, cmap='gray') 
    plt.show() 
 
def run_denoise(model, test_set):
    img, _ = test_set[1]
    noise = torch.rand(img.shape)
    
    noisy_img=img+noise
    
    img_batch = noisy_img.unsqueeze(0).view(1, -1)
    
    # Run the model
    with torch.no_grad():
        output_batch = model(img_batch)
    
    # Unflatten the input and output to visualize it as an image
    noisy_img = noisy_img.squeeze().numpy()
    output_img = output_batch.view(1, 28, 28).squeeze().numpy()
    
    f = plt.figure() 
    f.add_subplot(1,2,1) 
    plt.imshow(noisy_img, cmap='gray') 
    f.add_subplot(1,2,2) 
    plt.imshow(output_img, cmap='gray') 
    plt.show() 

def run_lin_interp(model, test_set, n_steps = 8):
    
    img1, _ = test_set[0]
    img2, _ = test_set[1]
    
    # Step 1: Encode the images to get the bottleneck tensors
    z1 = model.encode(img1.unsqueeze(0).view(1, -1))
    z2 = model.encode(img2.unsqueeze(0).view(1, -1))

    # Step 2: Perform linear interpolation
    alphas = torch.linspace(0, 1, n_steps)
    interpolated_images = []

    for alpha in alphas:
        z_int = alpha * z1 + (1 - alpha) * z2
        img_int = model.decode(z_int)
        interpolated_images.append(img_int)
    
    # Step 3: Plotting the interpolated images
    plt.figure(figsize=(20, 4))

    for i, img in enumerate([img2] + interpolated_images + [img1]):
        plt.subplot(1, n_steps + 2, i + 1)
        plt.imshow(img.detach().numpy().reshape(28, 28), cmap='gray')

    plt.show()
    

def main(l):
    model = MLP()
    model.load_state_dict(torch.load(l))
    model.eval()
    
    test_transform = transforms.Compose([transforms.ToTensor()]) 
    test_set = datasets.MNIST('./data/mnist', train=False, download=False, transform=test_transform)
    
    run_test(model, test_set)
    run_denoise(model, test_set)
    run_lin_interp(model, test_set)

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Training script")

    # Add arguments
    parser.add_argument("-l", type=str, required=True, help="The l argument")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.l)