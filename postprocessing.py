# WILD (Wildlife Low-cost Data-labelling), unreleased
"""
Post-processing utils to refine pseudo-labels for object detection on aerial images (conservation-focused)

Usage - tasks:

1. Filtering of small noisy bounding boxes

    $ python postprocessing.py \
        --task "filt" \
        --image_list path_to_image_list \
        --source_path path_to_pseudo_labels_before_processing \
        --dest_path path_to_pseudo_labels_after_processing  \
        --param_value filt_thresh


2. Padding of bounding boxes

    $ python postprocessing.py \
        --task "pad" \
        --image_list path_to_image_list  \
        --source_path path_to_pseudo_labels_before_processing \
        --dest_path path_to_pseudo_labels_after_processing \
        --param_value pad_value
     

3. Edge artefact filtering

    $ python postprocessing.py \
        --task "edge_filt" \
        --image_list path_to_image_list \
        --source_path path_to_pseudo_labels_before_processing" \
        --dest_path path_to_pseudo_labels_after_processing \
        --im_size 640


Tutorial:   WILD_Self_Supervised_Pipeline.ipynb, WILD_Unsupervised_Pipeline.ipynb
"""


import os
import argparse

def batched_v_small_obj_filtering(image_list, dir, output_dir, filter_threshold):
    """
    Filter out small noisy bounding boxes.

    Arguments:
    - Text file with the filenames of the images
    - Path to input labels
    - Path to output directory
    - Filtering threshold

    """

    if(image_list):
      f = open(image_list, "r")
      filenames = f.readlines()
    i = 0
    for filename in filenames:
        text_filename = filename[:-5] + ".txt"
        if(os.path.exists(dir + text_filename)):
            f = open(dir + text_filename,"r")
            lines = f.readlines()
            f.close()
            newlines = []
            for line in lines:
                info = line.split(" ")
                if((float(info[3]) > filter_threshold) | (float(info[4]) > filter_threshold)):
                    newlines.append(line)
                else:
                    continue

            if(len(newlines)!=0):
                f = open(output_dir + text_filename, "w")
                f.write("".join(newlines))
                f.close()
            else:
                f = open(output_dir + text_filename, "w")
                f.write("")
                f.close()
        i+=1



def pad_boxes(image_list, source_path, dest_path, pad_value):
    """
    Pad bounding boxes for clearer training targets.

    Arguments:
    - Text file with the filenames of the images
    - Path to input labels
    - Path to output directory
    - Padding value

    """

    if(image_list):
      f = open(image_list, "r")
      filenames = f.readlines()
    print("Pad bounding boxes for " + str(len(filenames)) + " labels.")
    for file in filenames:
      if os.path.exists(source_path + file[0:-5] + ".txt"):
        f = open(source_path + file[0:-5] + ".txt", "r")
        lines = f.readlines()
        newlines = []
        for line in lines:
            info = line.split(" ")
            padded_width = float(info[3]) + pad_value
            info[3] = str(1.0) if (padded_width > 1.0) else str(padded_width)
            padded_height = float(info[4][:-1]) + pad_value
            info[4] = str(1.0) if (padded_height > 1.0) else str(padded_height)
            newlines.append(" ".join(info))
        f.close()

        if(len(newlines)!=0):
            f = open(dest_path + file[0:-5] + ".txt", "w")
            f.write("\n".join(newlines))
            f.close()




def batched_edge_artefact_filtering(label_dir, dest_path, im_width_and_height):
    """
    Filter out edge artefacts (common in anomaly masks).

    Arguments:
    - Path to input labels
    - Path to output directory
    - Image width and height (assumed equal)

    """
    
    files = os.listdir(label_dir)

    i = 0
    for filename in files:
        if(i%100 == 0):
            print(i)
        if(filename.endswith(".txt")):
            f = open(label_dir + filename,"r")
            lines = f.readlines()
            f.close()
            newlines = []
            for line in lines:
                split_line = line.split(" ")
                x_centre = float(split_line[1])*im_width_and_height
                y_centre = float(split_line[2])*im_width_and_height
                w = float(split_line[3])*im_width_and_height
                h = float(split_line[4])*im_width_and_height
                # calculate ymin, ymax, xmin and xmax
                xmin = int(x_centre - (0.5*w))
                ymin = int(y_centre - (0.5*h))
                xmax = int(x_centre + (0.5*w))
                ymax = int(y_centre + (0.5*h))
                #look for border pixels
                if not ( (xmin == 0) | (ymin == 0)  | (xmax == 0) | (ymax == 0)):
                    if not ((xmin == im_width_and_height) | (ymin == im_width_and_height)  | (xmax == im_width_and_height) | (ymax == im_width_and_height)):
                        newlines.append(line)
                    else:
                        continue
                else:
                    continue 

            if(len(newlines)!=0):
                f = open(dest_path + filename, "w")
                f.write("".join(newlines))
                f.close()
            i+=1





def parse_args(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, default='', help='Text file containing all the filenames of the images to be labelled')
    parser.add_argument('--source_path', type=str, default='', help='Path to input labels')
    parser.add_argument('--dest_path', type=str, default='', help='Path to output directory to which processed labels will be written')
    parser.add_argument('--task', type=str, default='', help='Postprocessing task to be performed')
    parser.add_argument('--param_value', type=float, default=0, help='Parameter value applicable to task')
    parser.add_argument('--im_size', type=int, default=640, help='Image width/height')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()



def main(args):
   
   if(args.task == 'filt'):
       print("Filtering small bounding boxes/noise...")
       print("Chosen filtering threshold: " + str(args.param_value))
       batched_v_small_obj_filtering(args.image_list, args.source_path, args.dest_path, args.param_value)
   
   elif (args.task == 'pad'):
       print("Padding bounding boxes for better representation...")
       print("Chosen padding value: " + str(args.param_value))
       pad_boxes(args.image_list, args.source_path, args.dest_path, args.param_value)

   elif (args.task == 'edge_filt'):
       print("Filter out edge artefacts...")
       batched_edge_artefact_filtering(args.source_path, args.dest_path, args.im_size)
   
   else:
      print("Invalid post-processing task supplied.")



if __name__ == '__main__':
    args = parse_args()
    main(args)