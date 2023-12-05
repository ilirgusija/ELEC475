from torch.utils.data import Dataset
import torch
from torchvision import transforms
import re
import cv2
import os
import numpy as np
from PIL import Image
import torch
import sys

def generate_gaussian_heatmap(centre, image_size):
    # Create a grid of (x,y) coordinates
    x = np.arange(0, image_size[1], 1, float)
    y = np.arange(0, image_size[0], 1, float)
    y = y[:, np.newaxis]

    sigma = 7
    # Calculate the 2D Gaussian
    heatmap = np.exp(-((x - centre[0])**2 + (y - centre[1])**2) / (2 * sigma**2))

    return heatmap

class HeatmapKeypointDataset(Dataset):
    def __init__(self, root_dir, labels_file, target_size=(256, 256)):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.target_size = target_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.labels_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                match = re.match(r'([^,]+),"\((\d+),\s*(\d+)\)"', line.strip())
                if match:
                    image_name, x_str, y_str = match.groups()
                    image_path = os.path.join(self.root_dir, , image_name.strip())
                    x = float(x_str)
                    y = float(y_str)
                    keypoint = [x, y]
                    data.append((image_path, keypoint))
                else:
                    print("Invalid line format or unable to parse:", line.strip())

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, keypoints = self.data[idx]
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image:", image_path)
            return None, None

        # Scale keypoints to the range [0, 1]
        scale_x = image.shape[1]
        scale_y = image.shape[0]
        keypoints[0] = keypoints[0] / scale_x
        keypoints[1] = keypoints[1] / scale_y

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

        image_resized = transform(image)

        heatmap = np.zeros(self.target_size, dtype=np.float32)
        keypoint_x_scaled = int(keypoints[0] * self.target_size[1])
        keypoint_y_scaled = int(keypoints[1] * self.target_size[0])
        heatmap = generate_gaussian_heatmap((keypoint_x_scaled, keypoint_y_scaled), self.target_size)

        heatmap = np.expand_dims(heatmap, axis=0)  # Change shape from (H, W) to (1, H, W)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return image_resized, heatmap


# not heatmap!
class KeypointDataset(Dataset):
    def __init__(self, root_dir, labels_file, target_size=(256, 256)):
        self.root_dir = root_dir
        self.labels_file = os.path.join(root_dir, labels_file)
        self.target_size = target_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.labels_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                match = re.match(r'([^,]+),"\((\d+),\s*(\d+)\)"', line.strip())
                if match:
                    image_name, x_str, y_str = match.groups()
                    image_path = os.path.join(self.root_dir, image_name.strip())
                    x = float(x_str)
                    y = float(y_str)
                    keypoint = [x, y]
                    data.append((image_path, keypoint))
                else:
                    print("Invalid line format or unable to parse:", line.strip())

        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, keypoints = self.data[idx]
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image:", image_path)
            return None, None

        scale_x = image.shape[1] / 255
        scale_y = image.shape[0] / 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints[0] = keypoints[0] / scale_x
        keypoints[1] = keypoints[1] / scale_y

        image = Image.fromarray(image)
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0., std=1.)])
        image = transform(image)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)/255.

        return image, keypoints