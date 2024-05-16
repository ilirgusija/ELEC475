import torch
import os
import cv2
import argparse
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from KittiDataset import KittiDataset, batch_ROIs, minibatch_ROIs
from KittiAnchors import Anchors
from torchvision import transforms
from classifier import object_classifier, backends
from yoda import YODA
import cv2
import matplotlib.pyplot as plt
import torchvision
import sys
import numpy as np

# def display_images_with_boxes(i, save_dir, image, bounding_boxes):
#     """
#     Display images with bounding boxes.

#     :param image_paths: List of paths to the images.
#     :param bounding_boxes: List of bounding boxes for each image. Each bounding box is a tuple (pt1, pt2) where pt1 and pt2 are coordinates of the box corners.
#     :param display_count: Number of images to display.
#     """
#     # Load the image
#     # image = cv2.imread(image_path)

#     if image.dtype != np.uint8:
#             image = (image * 255).astype(np.uint8)

#     # Draw each bounding box
#     for box in bounding_boxes:
#         pt1 = (int(box[0][1]), int(box[0][0]))
#         pt2 = (int(box[1][1]), int(box[1][0]))
#         cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

#     # Save the image
#     save_path = f"{save_dir}/processed_image_{i}.jpg"
#     cv2.imwrite(save_path, image)
#     print(f"Saved image {i} to {save_path}")


def main(model_type, data_dir, classifier_params, device):
    dataset = KittiDataset(data_dir, training=False)
    save_dir="../output"
    if model_type=='resnet_18':
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))
    else:
        model = object_classifier(encoder=getattr(backends, model_type, None))
        model = object_classifier(encoder=getattr(backends, 'encoder', None))

    model.load_state_dict(torch.load(classifier_params))
    anchors = Anchors()
    model.to(device)
    # Iterating through batches
    total_iou_scores = []  # To store IoU scores of all 'Car' ROIs  
    display_count = 0  # Counter to keep track of the number of images displayed
    
    for i, (image, label) in enumerate(dataset):
        # label = [float(x) for x in label]
        # print(label)
        # sys.exit()
        # print(type(image))
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

        # Transform ROIs into a batch
        batched_ROIs = batch_ROIs(ROIs, shape=(150, 150))

        # Move batch to device and pass through the classifier
        batched_ROIs = batched_ROIs.to(device)
        predictions = model(batched_ROIs)

        for k, pred in enumerate(predictions):
            is_car = (pred >= 0.5).float()
            if is_car:
                iou_score = anchors.calc_max_IoU(boxes[k], car_ROIs)
                total_iou_scores.append(iou_score)
                box = boxes[k]
                pt1 = (int(box[0][1]), int(box[0][0]))
                pt2 = (int(box[1][1]), int(box[1][0]))
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        # Display the image with bounding boxes for the first two images only
        if display_count < 5:
            # Save the image
            save_path = f"{save_dir}/processed_image_{i}.jpg"
            cv2.imwrite(save_path, image)
            print(f"Saved image {i} to {save_path}")
            display_count += 1
        
        print(f"Processed image {i}")
        

    mean_iou = sum(total_iou_scores) / len(total_iou_scores) if total_iou_scores else 0
    print(f"Mean IoU for 'Car' ROIs: {mean_iou}")


if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the parent directory (assuming the script is in 'src')
    parent_dir = os.path.dirname(current_script_dir)

    # Set the default data directory
    default_data_dir = os.path.join(parent_dir, "data/Kitti8")
    default_classifier = os.path.join(parent_dir, "output/classifier.pth")
    
    print('running YODA ...')

    label_file = 'labels.txt'

    argParser = argparse.ArgumentParser(description="Script for running YODA model")
    argParser.add_argument('-m', '--model_type', choices=['resnet', 'se_resnet', 'se_resneXt', 'resnet_18'], default='resnet_18', help='Classifier type')
    argParser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    argParser.add_argument('-c', '--classifier_params', type=str, default=default_classifier, help='Classifier weights')
    argParser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')

    args = argParser.parse_args()

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cpu'
    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)
    
    main(args.model_type, args.data_dir, args.classifier_params, device)