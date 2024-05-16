import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import random
from collections import defaultdict, Counter

class BinaryROIsDataset(Dataset):
    def __init__(self, data_dir, data_type='train', transform=None, balance_data=False):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform
        self.img_dir = os.path.join(data_dir, data_type)
        self.labels = self._read_labels(os.path.join(self.img_dir, 'labels.txt'))

        if balance_data:
            self.labels = self._balance_data(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _read_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        return [(label[0], int(label[1])) for label in labels]

    def _count_labels(self):
        labels = [label for _, label in self.labels]
        return Counter(labels)

    def _balance_data(self, labels):
        # Separate labels by class
        class_0_labels = [label for label in labels if label[1] == 0]
        class_1_labels = [label for label in labels if label[1] == 1]

        # Truncate the majority class
        minority_class_size = min(len(class_0_labels), len(class_1_labels))
        balanced_class_0_labels = random.sample(class_0_labels, minority_class_size)
        balanced_class_1_labels = random.sample(class_1_labels, minority_class_size)

        # Combine and shuffle
        balanced_labels = balanced_class_0_labels + balanced_class_1_labels
        random.shuffle(balanced_labels)

        return balanced_labels