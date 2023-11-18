print('running ...')

import torch
import os
import cv2
import argparse
from KittiDataset import KittiDataset
from KittiAnchors import Anchors

save_ROIs = True
car_count = 0
nocar_count = 0
max_examples_per_class = 10000

def main():

    print('running KittiToYoda ...')

    label_file = 'labels.txt'

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    argParser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    argParser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    args = argParser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    output_dir = None
    if args.o != None:
        output_dir = args.o

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    training = True
    if args.m == 'test':
        training = False

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cpu'
    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=training)
    anchors = Anchors()

    
    for i, (image, label) in enumerate(dataset):
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        # print(car_ROIs)
        # for idx in range(len(car_ROIs)):
            # print(ROIs[idx])

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))

        # for j in range(len(centers)):
        #     center = centers[j]
        #     for k in range(len(anchor_shapes)):
        #         anchor_shape = anchor_shapes[k]
        #         pt1 = (int(center[0]-anchor_shape[0]/2), int(center[1]-anchor_shape[1]/2))
        #         pt2 = (int(center[0]+anchor_shape[0]/2), int(center[1]+anchor_shape[1]/2))
        #         cv2.rectangle(image, pt1, pt2, (0, 255, 255))
        #
        #     cv2.imshow('image', image)
        #     key = cv2.waitKey(0)
        #     if key == ord('x'):
        #         break

        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # print(ROI_IoUs)

        
        for k in range(len(boxes)):
            filename = str(i) + '_' + str(k) + '.png'
            if ROI_IoUs[k] >= IoU_threshold:
                if car_count >= max_examples_per_class:
                    continue  # Skip if already have enough 'Car' examples
                name_class = 1
                name = 'Car'
                car_count += 1
            else:
                if nocar_count >= max_examples_per_class:
                    continue  # Skip if already have enough 'NoCar' examples
                name_class = 0
                name = 'NoCar'
                nocar_count += 1

            if save_ROIs:
                cv2.imwrite(os.path.join(output_dir, filename), ROIs[k])
            labels.append([filename, name_class, name])


        if show_images:
            cv2.imshow('image', image1)
        # key = cv2.waitKey(0)
        # if key == ord('x'):
        #     break

        if show_images:
            image2 = image1.copy()

            for k in range(len(boxes)):
                if ROI_IoUs[k] > IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1],box[0][0])
                    pt2 = (box[1][1],box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

                # print(ROI_IoUs[k])
                # box = boxes[k]
                # pt1 = (box[0][1],box[0][0])
                # pt2 = (box[1][1],box[1][0])
                # cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
                # cv2.imshow('boxes', image2)
                # key = cv2.waitKey(0)
                # if key == ord('x'):
                #     break

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

    #     for j in range(len(label)):
    #         name = label[j][0]
    #         name_class = label[j][1]
    #         minx = int(label[j][2])
    #         miny = int(label[j][3])
    #         maxx = int(label[j][4])
    #         maxy = int(label[j][5])
    #
    #         roi = image[miny:maxy,minx:maxx]
    #         # roi = cv2.resize(roi, (width,height))
    #
    #         if save_full_ROI == True:
    #             filename = str(i) + '.png'
    #             cv2.imwrite(os.path.join(output_dir,filename), roi)
    #             labels += [[filename, name_class, name]]
    #             # print(i, filename, name_class, name)
    #
    #         dy = maxy - miny + 1
    #         dx = maxx - minx + 1
    #         if anchors.min_range[0] < dy and anchors.min_range[1] < dx:
    #             for k in range(len(anchors.shapes)):
    #                 shape = anchors.shapes[k]
    #                 dy = int(((maxy - miny)-shape[0])/2)
    #                 dx = int(((maxx - minx)-shape[1])/2)
    #                 miny2 = miny + dy
    #                 maxy2 = maxy - dy
    #                 minx2 = minx + dx
    #                 maxx2 = maxx - dx
    #
    #                 # print('break 08: ', miny2, maxy2, minx2, maxx2)
    #                 if dx > 0 and dy > 0 and miny2 < maxy2 and minx2 < maxx2:
    #                     roi = image[miny2:maxy2, minx2:maxx2]
    #                     # roi = cv2.resize(roi, (width,height))
    #                     filename = str(i) + '_' + str(k) + '.png'
    #                     cv2.imwrite(os.path.join(output_dir, filename), roi)
    #                     labels += [[filename, name_class, name]]
    #
    #                     if show_images == True:
    #                         image1 = image.copy()
    #                         cv2.rectangle(image1, (minx,miny), (maxx, maxy), (0,0,255))
    #                         cv2.imshow('image', image1)
    #                         cv2.imshow('roi', roi)
    #
    #                         key = cv2.waitKey(0)
    #                         if key == ord('x'):
    #                             break
    #
        print(i)

        if car_count >= max_examples_per_class and nocar_count >= max_examples_per_class:
            break  # Stop if both classes have enough examples

    #
    # print(labels)
    #
    if save_ROIs == True:
        with open(os.path.join(output_dir, label_file), 'w') as f:
            for k in range(len(labels)):
                filename = labels[k][0]
                name_class = str(labels[k][1])
                name = labels[k][2]
                f.write(filename + ' ' + name_class + ' ' + name + '\n')
        f.close()


###################################################################

main()
