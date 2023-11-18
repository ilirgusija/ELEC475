import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2

class KittiDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.img_dir = os.path.join(dir, self.mode, 'image')
        self.label_dir = os.path.join(dir, self.mode, 'label')
        self.transform = transform
        self.num = 0
        self.img_files = []
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        self.max = len(self)

        # print('break 12: ', self.img_dir)
        # print('break 12: ', self.label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label_path = os.path.join(self.label_dir, filename+'.txt')
        labels_string = None

        with open(label_path) as label_file:
            labels_string = label_file.readlines()
        labels = []

        for i in range(len(labels_string)):
            lsplit = labels_string[i].split(' ')
            label = [lsplit[0], int(self.class_label[lsplit[0]]), float(lsplit[4]), float(lsplit[5]), float(lsplit[6]), float(lsplit[7])]
            labels += [label]
        return image, labels

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


    class_label = {'DontCare': 0, 'Misc': 1, 'Car': 2, 'Truck': 3, 'Van': 4, 'Tram': 5, 'Cyclist': 6, 'Pedestrian': 7,
                   'Person_sitting': 8}

    def strip_ROIs(self, class_ID, label_list):
        ROIs = []
        for i in range(len(label_list)):
            ROI = label_list[i]
            if ROI[1] == class_ID:
                pt1 = (int(ROI[3]),int(ROI[2]))
                pt2 = (int(ROI[5]), int(ROI[4]))
                ROIs += [(pt1,pt2)]
        return ROIs

def batch_ROIs(ROIs, shape):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(shape)
    ])

    batch = torch.empty(size=(len(ROIs),3,shape[0],shape[1]))
    # batch = torch.empty(size=(shape[0],shape[1],len(ROIs)))
    # print('break 55: ', batch.shape)
    # print('break 55.5: ', shape)
    # resize = torchvision.transforms.Resize(size=shape)
    for i in range(len(ROIs)):
        ROI = ROIs[i]
        # print('break 650: ', ROI.shape, ROI.dtype)
        # print(ROI)
        # ROI = torchvision.transforms.ToTensor(ROI)
        # ROI = torch.from_numpy(ROI)
        # print('break 56: ', ROI.shape, ROI.dtype)
        # ROI = resize(ROI)
        ROI = transform(ROI)
        ROI = torch.swapaxes(ROI,1,2)
        # print('break 57: ', ROI.shape)
        # batch[i,:,:] = ROI[:,:]
        # batch = torch.cat([batch, ROI], dim=0)
        # print('break 664: ', i, batch.shape, ROI.shape)
        batch[i] = ROI
        # print('break 665: ', i, batch.shape)
    return batch

def minibatch_ROIs(ROIs, boxes, shape, minibatch_size):
    minibatch = []
    minibatch_boxes = []
    min_idx = 0
    while min_idx < len(ROIs)-1:
        max_idx = min(min_idx + minibatch_size, len(ROIs))
        minibatch += [batch_ROIs(ROIs[min_idx:max_idx], shape)]
        minibatch_boxes += [boxes[min_idx:max_idx]]
        min_idx = max_idx + 1
    return minibatch, minibatch_boxes