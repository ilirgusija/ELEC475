from torch.utils.data import Dataset
import torch
from torchvision import transforms
import re
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from random import shuffle
import random
import torch

# heatmap
class HeatmapKeypointDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        for folder_name in os.listdir(self.root_dir):
            if folder_name != '103':
                folder_path = os.path.join(self.root_dir, folder_name)
                if os.path.isdir(folder_path):
                    txt_file_path = os.path.join(folder_path, "labels.txt")
                    if os.path.exists(txt_file_path):
                        with open(txt_file_path, "r") as file:
                            lines = file.readlines()
                            for line in lines[1:]:
                                elements = line.replace(
                                    '"', '').strip().split(",")
                                if all(element is not None for element in elements):
                                    image_path = os.path.join(
                                        folder_path, elements[0])
                                    if not os.path.exists(image_path):
                                        print("Could not find image:",
                                              image_path)
                                        continue
                                    x_match = re.search(
                                        r'\((\d+)', elements[1])
                                    y_match = re.search(
                                        r'(\d+)\)', elements[2])
                                    if x_match and y_match:
                                        x = float(x_match.group(1))
                                        y = float(y_match.group(1))
                                        keypoint = [x, y]
                                        data.append((image_path, keypoint))
                                    else:
                                        print("Could not parse keypoints:",
                                              elements[1], elements[2])
                                else:
                                    print("Could not parse line:", line)

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

        keypoints[0] = keypoints[0] / scale_x
        keypoints[1] = keypoints[1] / scale_y

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        transform = transforms.Compose([transforms.Resize(self.target_size),
                                        transforms.ToTensor()])

        image_resized = transform(image)

        heatmap = np.zeros(self.target_size, dtype=np.float32)
        keypoint_x, keypoint_y = keypoints
        keypoint_x_scaled = int(
            keypoint_x * (self.target_size[1] / image_resized.size(2)))
        keypoint_y_scaled = int(
            keypoint_y * (self.target_size[0] / image_resized.size(1)))
        heatmap[keypoint_y_scaled, keypoint_x_scaled] = 1.0

        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return image_resized, heatmap


# not heatmap!
class KeypointDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        for folder_name in os.listdir(self.root_dir):
            if folder_name != '103':
                folder_path = os.path.join(self.root_dir, folder_name)
                if os.path.isdir(folder_path):
                    txt_file_path = os.path.join(folder_path, "labels.txt")
                    if os.path.exists(txt_file_path):
                        with open(txt_file_path, "r") as file:
                            lines = file.readlines()
                            for line in lines[1:]:
                                elements = line.replace(
                                    '"', '').strip().split(",")
                                if all(element is not None for element in elements):
                                    image_path = os.path.join(
                                        folder_path, elements[0])
                                    if not os.path.exists(image_path):
                                        print("Could not find image:",
                                              image_path)
                                        continue
                                    x_match = re.search(
                                        r'\((\d+)', elements[1])
                                    y_match = re.search(
                                        r'(\d+)\)', elements[2])
                                    if x_match and y_match:
                                        x = float(x_match.group(1))
                                        y = float(y_match.group(1))
                                        keypoint = [x, y]
                                        data.append((image_path, keypoint))
                                    else:
                                        print("Could not parse keypoints:",
                                              elements[1], elements[2])
                                else:
                                    print("Could not parse line:", line)

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
