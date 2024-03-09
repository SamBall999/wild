# WILD (Wildlife Low-cost Data-labelling), unreleased
"""
Run WILD self-supervised or unsupervised pipeline to generate pseudo-labels for object detection on aerial images (conservation-focused)

Usage - sources:

1. Generate labels from self-supervised attention maps

    $ python autolabel.py \
        --input_type 'attention' \
        --img_path "./data/images/train" \
        --msk_path "./data/ss_labels/masks/train/" \
         --dest_path "./data/ss_labels/raw_labels/train/" \
        --image_list "./data/train_files.txt" \
        --batch_start_index 0 \
        --batch_size 100 \
        --sample_interval 1 \
        --threshold 80


2. Generate labels from unsupervised anomaly masks

    $ python autolabel.py \
        --input_type 'anomaly' \
        --img_path "./data/images/train" \
        --msk_path "./data/us_labels/masks/train/" \
        --dest_path "./data/us_labels/raw_labels/train/" \
        --image_list "./data/train_files.txt" \
        --batch_start_index 0 \
        --batch_size 100 \
        --sample_interval 1
    

Models:     Pipeline to create labels using attention maps from self-supervised DINO or anomaly masks from unsupervised FastFlow
Tutorial:   WILD_Self_Supervised_Pipeline.ipynb, WILD_Unsupervised_Pipeline.ipynb, 
"""


import os
import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.io import read_image
from skimage.measure import label, regionprops

plt.rcParams["savefig.bbox"] = 'tight'

def attention_map_to_segmentation_mask(attention_map_path, img_name, thr, save):
    """
    Convert attention map to binary segmentation mask.

    Arguments:
    - Path to attention map
    - Image name
    - Threshold
    - Whether to save output mask

    Returns:
    - Binary segmentation mask
    """
    # check if attention map exists
    if (os.path.exists(attention_map_path + img_name)):
      img = read_image(attention_map_path + img_name)
    else:
      f = open('missing.txt', 'a')
      f.write(img_name +"\n")
      f.close()
      return None
    # thresholding
    grey_img = transforms.Grayscale()(img[0:3, :, :])
    grey_img_thresholded = (grey_img > thr).to(grey_img.dtype)*255
    # save
    if (save==True):
      Image.fromarray(grey_img_thresholded[0].numpy()).save(os.path.join("map-" + img_name))
    return grey_img_thresholded[0].numpy()




def combine_masks(frame_name, dest_path, masks, save):
    """
    Combine six binary masks from the six attention heads using the AND operation.

    Arguments:
    - Image name
    - Destination path
    - Array of six binary masks
    - Whether to save the output mask

    Returns:
    - Combined mask
    """
    # add masks
    total_num_masks = 6
    result = masks[0] + masks[1] + masks[2] + masks[3] + masks[4] + masks[5]
    result /= total_num_masks
    result[result < 255] = 0
    # save results
    if (save==True):
      cv2.imwrite(dest_path + frame_name + '-mask.png', result)
    return result




def mask_to_bb(filename, mask, dest_path, im_width, im_height):
    """
    Convert binary mask to .txt label containing bounding boxes in YOLO format.

    Arguments:
    - Image filename
    - Corresponding combined binary mask
    - Destination path for output .txt label
    - Image width
    - Image height
    """

    # find connected components
    lbl_0 = label(mask, connectivity=2)
    props = regionprops(lbl_0)
    # remove file if it exists
    if os.path.exists(dest_path + filename + ".txt"):
      os.remove(dest_path + filename + ".txt")
    # write bounding boxes to file
    f = open(dest_path + filename + ".txt", "a")
    for prop in props:
        w = prop.bbox[3] - prop.bbox[1]
        h = prop.bbox[2] - prop.bbox[0]
        x_cen = prop.bbox[1]  + (0.5*w)
        y_cen = prop.bbox[0]  + (0.5*h)
        # normalise
        w = w/im_width
        h = h/im_height
        x_cen = x_cen/im_width
        y_cen = y_cen/im_height
        f.write('0 ' + str(x_cen) + " " + str(y_cen) + " " + str(w) + " " + str(h))
        f.write("\n")
    f.close()




def attention_maps_to_masks_to_bb(msk_path, dest_path, filename, threshold, image_size, save):
    """
    Pipeline to convert six attention maps to a .txt label containing bounding boxes in YOLO format.

    Arguments:
    - Path to attention maps
    - Path to output directory
    - Filename of image 
    - Threshold
    - Image size
    - Whether to save combined mask
    """
    attention_maps= []
    for i in range(0, 6):
      returned_map = attention_map_to_segmentation_mask(msk_path, filename + "-attn-head" + str(i) + ".png", threshold, save)
      if (returned_map is None):
        print("Missing attention map: ", filename + "-attn-head" + str(i) + ".png")
        return
      else:
        attention_maps.append(returned_map.astype("float32"))
    combined_mask = combine_masks(filename, msk_path, attention_maps, save)
    mask_to_bb(filename, combined_mask, dest_path, image_size, image_size)




def batched_attention_maps_to_bb_final(img_path, msk_path, dest_path, image_list, batch_start_index, batch_size, sample_interval, threshold, image_size):
    """
    Creates pseudo-labels from attention maps for a batch of input images.

    Arguments:
    - Path to images
    - Path to attention maps
    - Path to output directory
    - Textfile with names of all images
    - Batch start index
    - Batch size
    - Sample interval
    - Threshold at which binary thresholding is performed
    - Image size
    """
    if(image_list):
        f = open(image_list, "r")
        filenames = f.readlines()
    else:
        filenames = sorted(os.listdir(img_path))
    for i in range(batch_start_index, batch_start_index + (batch_size*sample_interval), sample_interval):
        if(image_list):
            attention_maps_to_masks_to_bb(msk_path, dest_path, filenames[i][0:-5], threshold, image_size, False)
        else:
            attention_maps_to_masks_to_bb(msk_path, dest_path, filenames[i][:-4], threshold, image_size, False)





def heatmaps_to_masks_to_bb(msk_path, dest_path, filename, threshold, image_size, save):
    """
    Pipeline to convert an anomaly detection heatmap to a .txt label containing bounding boxes in YOLO format.

    Arguments:
    - Path to anomaly detection heatmaps
    - Path to output directory
    - Filename of image 
    - Threshold
    - Image size
    - Whether to save combined mask
    """
    returned_map = attention_map_to_segmentation_mask(msk_path, filename + ".png", threshold, save)
    if (returned_map is None):
        print("Missing attention map: ", filename + ".png")
        return
    mask_to_bb(filename, returned_map.astype("float32"), dest_path, image_size, image_size)





def segmentation_masks_to_bb(msk_path, dest_path, filename, image_size):
    """
    Pipeline to convert an anomaly detection segmentation mask to a .txt label containing bounding boxes in YOLO format.

    Arguments:
    - Path to anomaly segmentation masks
    - Path to output directory
    - Filename of image 
    - Image size
    """
    segmentation_mask = cv2.imread(msk_path + filename + ".png")
    if (segmentation_mask is None):
        print("Missing anomaly mask: ", filename + ".png")
        return
    mask_to_bb(filename, segmentation_mask[:, :, 0], dest_path, image_size, image_size)





def batched_anomaly_masks_to_bb_final(img_path, msk_path, dest_path, image_list, batch_start_index, batch_size, sample_interval, image_size):
    """
    Creates pseudo-labels from anomaly detection segmentation masks for a batch of input images.

    Arguments:
    - Path to images
    - Path to attention maps
    - Path to output directory
    - Textfile with names of all images
    - Batch start index
    - Batch size
    - Sample interval
    - Image size
    """
    if(image_list):
        f = open(image_list, "r")
        filenames = f.readlines()
    else:
        filenames = sorted(os.listdir(img_path))
    for i in range(batch_start_index, batch_start_index + (batch_size*sample_interval), sample_interval):
        if(image_list):
            segmentation_masks_to_bb(msk_path, dest_path, filenames[i][0:-5], image_size)
        else:
            segmentation_masks_to_bb(msk_path, dest_path, filenames[i][:-4], image_size)





def parse_args(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, default='', help='Input source to convert to labels: attention maps or anomaly masks')
    parser.add_argument('--img_path', type=str, default='', help='Path to images')
    parser.add_argument('--msk_path', type=str, default='', help='Path to attention maps')
    parser.add_argument('--dest_path', type=str, default='', help='Path to output directory to which labels will be written')
    parser.add_argument('--image_list', type=str, default='', help='Text file containing all the filenames of the images to be labelled')
    parser.add_argument('--batch_start_index', type=int, default=0, help='Index from which to start labelling')
    parser.add_argument('--batch_size', type=int, default=100, help='Total number of images in set')
    parser.add_argument('--sample_interval', type=int, default=1, help='Controls whether a sample of the set is labelled (default to every image in set)')
    parser.add_argument('--threshold', type=int, default=80, help='Lower thresholds recall more objects but increase noise, higher thresholds decrease noise but may miss relevant objects')
    parser.add_argument('--image_size', type=int, default=640, help='Size of images and masks, expected that height=width')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()



def main(args):

    if (args.input_type == 'attention'):
        print("Generating pseudo-labels based on attention maps...") 
        batched_attention_maps_to_bb_final(args.img_path, args.msk_path, args.dest_path, args.image_list, args.batch_start_index, args.batch_size, args.sample_interval, args.threshold, args.image_size)

    elif (args.input_type == 'anomaly'):
        print("Generating pseudo-labels based on anomaly segmentation masks...") 
        batched_anomaly_masks_to_bb_final(args.img_path, args.msk_path, args.dest_path, args.image_list, args.batch_start_index, args.batch_size, args.sample_interval, args.image_size)
    
    else: 
        print("Invalid input type given. Please choose attention or anomaly input type.")


if __name__ == '__main__':
    args = parse_args()
    main(args)