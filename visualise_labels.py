# WILD (Wildlife Low-cost Data-labelling), unreleased
"""
Additional WILD utils including visualisation of generated pseudo-labels and mask IOU calculation between pseudo-labels and ground truth examples

Usage - single vs multi:

1. Visualise bounding boxes for single images

    $ python visualise_labels.py \
        --filename image_name \
        --img_path path_to_images \
        --gt_label_path path_to_gt_labels \
        --pseudo_label_path path_to_pseudo_labels \
        --output_dir path_to_output_directory \
        --image_size image_size

        
2. Visualise bounding boxes for multiple images

    $ python visualise_labels.py \
        --multi \
        --image_list path_to_image_list \
        --img_path path_to_images \
        --gt_label_path path_to_gt_labels \
        --pseudo_label_path path_to_pseudo_labels \
        --output_dir path_to_output_directory \
        --image_size image_size


Usage - sources:

1. Visualise bounding boxes for pseudo-labels only

    $ python visualise_labels.py \
        --multi \
        --image_list path_to_image_list \
        --img_path path_to_images \
        --gt_label_path '' \
        --pseudo_label_path path_to_pseudo_labels \
        --output_dir path_to_output_directory \
        --image_size image_size

        
2. Visualise bounding boxes for pseudo-labels vs ground truth

    $ python visualise_labels.py \
        --multi \
        --image_list path_to_image_list \
        --img_path path_to_images \
        --gt_label_path path_to_gt_labels \
        --pseudo_label_path path_to_pseudo_labels \
        --output_dir path_to_output_directory \
        --image_size image_size

        
Tutorial:   WILD_Self_Supervised_Pipeline.ipynb, WILD_Unsupervised_Pipeline.ipynb
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_image


def plot(imgs):
    """
    Visualise images.

    Arguments:
    - Images to be plotted

    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.show()



def simple_resize(img_name, image_size, inter = cv2.INTER_AREA, colour=False):
    """
    Resizes input image to specified width and height.

    Arguments:
    - Path to image
    - Desired image size (assumes width = height)
    - Method of interpolation
    - Boolean specifying if image is colour

    Returns:
    - Resized image
    """

    im = cv2.imread(img_name)
    resized = cv2.resize(im, (image_size, image_size), interpolation = inter)

    # threshold
    if(colour==False):
        resized[resized < 127] = 0
        resized[resized >= 127] = 255

    return resized



def bb_to_mask(filepath_and_name, im_width, im_height, save):
    """
    Convert a .txt label with bounding boxes in YOLO format to a binary mask.

    Arguments:
    - Filepath to .txt label
    - Image width
    - Image height
    - Whether to save output mask or not

    Returns:
    - Binary mask showing positions of bounding boxes in frame
    """

    mask = np.zeros((im_width, im_height),dtype=np.uint8) # initialize mask

    # if annotations, add white blocks, otherwise plain black
    if(os.path.exists(filepath_and_name + ".txt")):
      f = open(filepath_and_name + ".txt","r")
      lines = f.readlines()
      # need to multiply by image width and height
      for line in lines:
          line = line.split(" ")
          x_centre = float(line[1])*im_width
          y_centre = float(line[2])*im_height
          w = float(line[3])*im_width
          h = float(line[4])*im_height
          # calculate ymin, ymax, xmin and xmax
          xmin = int(x_centre - (0.5*w))
          ymin = int(y_centre - (0.5*h))
          xmax = int(x_centre + (0.5*w))
          ymax = int(y_centre + (0.5*h))
          mask[ymin:ymax,xmin:xmax] = 255 # fill with white pixels

    if(save==True):
      cv2.imwrite( filepath_and_name.split("/")[-1] + '_gt_seg_mask.png', mask) # save mask

    return mask


def calc_binary_mask_iou(img_gt, img_pseudo):
    """
    Calculate IOU between two binary masks.

    Arguments:
    - First binary mask
    - Second binary mask

    Returns:
    - Calculated IOU value
    """

    mask1 = np.asarray(img_gt)
    mask2 = np.asarray(img_pseudo)
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and( mask1==255,  mask2==255 ))
    if intersection == 0:
        return 0.0
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou



def plot_bb(filename, output_dir, gt_filename, pseudo_filename, resized):
    """
    Plots bounding boxes from .txt label onto resized image for visualisation.

    Arguments:
    - Filename and path of .txt ground truth label
    - Filename and path of .txt pseudo label

    Note: Only one of gt_filename or pseudo_filename is required i.e. can plot bounding boxes for just ground truth label or just pseudo label instead of both.
    """

    img_1 = resized.copy()
    im_width = resized.shape[1]
    im_height = resized.shape[0]

    if(os.path.exists(gt_filename + ".txt")):
      f = open(gt_filename + ".txt","r")
      lines = f.readlines()
      for line in lines:
          line = line.split(" ")
          x_centre = float(line[1])*im_width
          y_centre = float(line[2])*im_height
          w = float(line[3])*im_width
          h = float(line[4])*im_height
          # calculate ymin, ymax, xmin and xmax
          xmin = int(x_centre - (0.5*w))
          ymin = int(y_centre - (0.5*h))
          xmax = int(x_centre + (0.5*w))
          ymax = int(y_centre + (0.5*h))
          cv2.rectangle(img_1, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
      f.close()

    if(os.path.exists(pseudo_filename + ".txt")):
      f1 = open(pseudo_filename + ".txt","r")
      lines = f1.readlines()
      for line in lines:
          line = line.split(" ")
          x_centre = float(line[1])*im_width
          y_centre = float(line[2])*im_height
          w = float(line[3])*im_width
          h = float(line[4])*im_height
          # calculate ymin, ymax, xmin and xmax
          xmin = int(x_centre - (0.5*w))
          ymin = int(y_centre - (0.5*h))
          xmax = int(x_centre + (0.5*w))
          ymax = int(y_centre + (0.5*h))
          cv2.rectangle(img_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite(output_dir + filename[0:-4] + '_bbs.png', img_1) # save mask



def calc_iou_and_plot_bb(filename, image_path, gt_label_path, pseudo_label_path, output_dir, image_size):
    """
    Calculate IOU between two binary masks and plot bounding boxes onto image.

    Arguments:
    - Image filename
    - Path to input image
    - Path to ground truth labels
    - Path to pseudo-labels
    - Output directory for bounding box visualisation
    - Image size
    """

    # Calculate IOU
    img_gt = bb_to_mask(gt_label_path + filename[0:-4], image_size, image_size, False) # normal
    img_pseudo = bb_to_mask(pseudo_label_path + filename[0:-4], image_size, image_size, False) # pseudo
    iou = calc_binary_mask_iou(img_gt, img_pseudo)
    print("Binary Mask IOU between ground truth and pseudolabel: " + str(iou))

    # Plot bounding boxes on resized image
    resized = simple_resize(image_path + filename, image_size, cv2.INTER_AREA, True)
    plot_bb(filename, output_dir, gt_label_path + filename[0:-4], pseudo_label_path + filename[0:-4], resized)



def batch_plot_bb(image_list, image_path, gt_label_path, pseudo_label_path, output_dir, image_size):
    """
    Calculate IOU between two binary masks and plot bounding boxes onto image.

    Arguments:
    - Image filename
    - Path to input image
    - Path to ground truth labels
    - Path to pseudo-labels
    - Output directory for bounding box visualisations
    - Image size
    """

    if(image_list):
        f = open(image_list, "r")
        filenames = f.readlines()
    else:
        filenames = sorted(os.listdir(image_path))

    # Plot bounding boxes on resized images in set
    for file in filenames:
        filename = file[:-1]
        resized = simple_resize(image_path + filename, image_size, cv2.INTER_AREA, True)
        plot_bb(filename, output_dir, gt_label_path + filename[0:-4], pseudo_label_path + filename[0:-4], resized)



def parse_args(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--multi', nargs='?', const=True, default=False, help='Visualise bounding boxes for set of multiple images')
    parser.add_argument('--filename', type=str, default='', help='Filename if visualising a single image')
    parser.add_argument('--image_list', type=str, default='', help='Text file containing all the filenames of the images to be labelled')
    parser.add_argument('--img_path', type=str, default='', help='Path to images')
    parser.add_argument('--gt_label_path', type=str, default='', help='Path to ground truth labels')
    parser.add_argument('--pseudo_label_path', type=str, default='', help='Path to pseudo labels')
    parser.add_argument('--output_dir', type=str, default='', help='Path to directory where visualisations will be saved')
    parser.add_argument('--image_size', type=int, default=640, help='Size of images and masks, expected that height=width')

    return parser.parse_known_args()[0] if known else parser.parse_args()



def main(args):
   
   if(args.multi):
       print("Visualising bounding boxes for image set...")
       batch_plot_bb(args.image_list, args.img_path, args.gt_label_path, args.pseudo_label_path, args.output_dir, args.image_size)
   
   else:
      print("Visualising bounding boxes for specified image...")
      calc_iou_and_plot_bb(args.filename, args.img_path, args.gt_label_path, args.pseudo_label_path, args.output_dir, args.image_size)



if __name__ == '__main__':
    args = parse_args()
    main(args)