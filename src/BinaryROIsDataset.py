import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2

class BinaryROIsDataset(Dataset):
    def __init__(self, data_dir, data_type='train', transform=None, padding=True):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform
        self.padding = padding
        self.img_dir = os.path.join(data_dir, data_type)
        self.labels = self._read_labels(os.path.join(data_dir, f'labels.txt'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if self.padding:
            image = self._pad_image(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

    def _read_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        return [(label[0], int(label[1])) for label in labels]

    def _pad_image(self, image):
        old_size = image.shape[:2] # old_size is in (height, width) format
        desired_size = 150

        # Calculate the new size, maintaining the aspect ratio
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # Resize the image to new_size
        resized = cv2.resize(image, (new_size[1], new_size[0]))

        # Create a new image and paste the resized on it
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im