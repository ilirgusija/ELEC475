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

car_count = 0
nocar_count = 0
max_examples_per_class = 10000    

def run(model, dataset, device):
    # Iterating through batches
    total_iou_scores = []  # To store IoU scores of all 'Car' ROIs  
    display_count = 0  # Counter to keep track of the number of images displayed
    
    for i, (image, label) in enumerate(dataset):
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        
        anchor_centers = model.anchors.calc_anchor_centers(image.shape, model.anchors.grid)
        ROIs, boxes = model.anchors.get_anchor_ROIs(image, anchor_centers, model.anchors.shapes)

        # Transform ROIs into a batch
        batched_ROIs = batch_ROIs(ROIs, shape=(150, 150))

        # Move batch to device and pass through the classifier
        batched_ROIs = batched_ROIs.to(device)
        predictions = model(batched_ROIs)

        for k, pred in enumerate(predictions):
            is_car = pred.argmax() == 1  # Assuming binary classification: [0, 1] where 1 is 'Car'
            if is_car:
                iou_score = model.anchors.calc_max_IoU(boxes[k], car_ROIs)
                total_iou_scores.append(iou_score)
                box = boxes[k]
                pt1 = (int(box[0][1]), int(box[0][0]))
                pt2 = (int(box[1][1]), int(box[1][0]))
                cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=2)

        # Display the image with bounding boxes for the first two images only
        if display_count < 2:
            cv2.imshow(f'Image {i}', image)
            key = cv2.waitKey(0)  # Wait for a key press to move to the next image
            if key == ord('q'):  # Press 'q' to quit the display loop early
                break
            display_count += 1

        print(f"Processed image {i}")

    mean_iou = sum(total_iou_scores) / len(total_iou_scores) if total_iou_scores else 0
    print(f"Mean IoU for 'Car' ROIs: {mean_iou}")

    
def main(data_dir, classifier_params, device):
    dataset = KittiDataset(data_dir, training=False)
    
    model = YODA(classifier_params)
    
    run(model, dataset, device)


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
    argParser.add_argument('-d', '--data_dir', type=str, default=default_data_dir, help='Data directory location')
    argParser.add_argument('-c', '--classifier_params', type=str, default=default_classifier, help='Classifier weights')
    argParser.add_argument('-cuda', choices=['Y', 'N'], default='Y', help='Whether to use CUDA (Y/N)')

    args = argParser.parse_args()

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cpu'
    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)
    
    main(args.data_dir, args.classifier_params, device)