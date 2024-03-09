## WILD (Wildlife Low-cost Data-labelling)

### Exploring Unsupervised and Self-Supervised Learning for Low-cost Data Annotation

Run WILD self-supervised or unsupervised pipeline to generate pseudo-labels for object detection on aerial images (conservation-focused)

### 1. Self-supervised pipeline

```
!python autolabel.py \
 --input_type 'attention' \
 --img_path "./data/images/train" \
 --msk_path "./data/ss_labels/masks/train/" \
 --dest_path "./data/ss_labels/raw_labels/train/" \
 --image_list "./data/train_files.txt" \
 --batch_start_index 0 \
 --batch_size 100 \
 --sample_interval 1 \
 --threshold 80
```

### 2. Unsupervised pipeline

```
!python autolabel.py \
 --input_type 'anomaly' \
 --img_path "./data/images/train" \
 --msk_path "./data/us_labels/masks/train/" \
 --dest_path "./data/us_labels/raw_labels/train/" \
 --image_list "./data/train_files.txt" \
 --batch_start_index 0 \
 --batch_size 100 \
 --sample_interval 1
```

### Visualise bounding boxes for generated labels

1. Visualise bounding boxes for multiple images

```
!python visualise_labels.py \
 --multi \
 --image_list "./data/train_files.txt" \
 --img_path "./data/images/train/" \
 --gt_label_path "./data/labels/train/" \
 --pseudo_label_path "./data/ss_labels/raw_labels/train/" \
 --output_dir "./data/ss_labels/bbs/train/" \
 --image_size 640
```

2. Visualise bounding boxes for single image

```
!python visualise_labels.py \
 --filename "t7-frame_020720.png" \
 --img_path "./data/images/train/" \
 --gt_label_path "./data/labels/train/" \
 --pseudo_label_path "./data/ss_labels/raw_labels/train/" \
 --output_dir "./data/ss_labels/bbs/train/" \
 --image_size 640
```

### Post-processing options

1. Filtering of noisy bounding boxes

```
!python postprocessing.py \
 --task "filt" \
 --image_list "./data/train_files.txt" \
 --source_path "./data/ss_labels/raw_labels/train/" \
 --dest_path "./data/ss_labels/processed_labels/train/" \
 --param_value 0.015
```

2. Padding of bounding boxes for clearer targets

```
!python postprocessing.py \
 --task "pad" \
 --image_list "./data/train_files.txt" \
 --source_path "./data/ss_labels/processed_labels/train/" \
 --dest_path "./data/ss_labels/processed_labels/train/" \
 --param_value 0.01
```

3. Edge artefact filtering

```
!python postprocessing.py \
 --task "edge_filt" \
 --image_list "./data/train_files.txt" \
 --source_path "./data/ss_labels/processed_labels/train/" \
 --dest_path "./data/ss_labels/processed_labels/train/" \
 --im_size 640
```

### Features

- Stream-lined repository for easy-access to the techniques described in the paper.
- All complexities have been abstracted away to allow for an accesible means of generating pseudo-labels through the interactive notebooks.
- Set up for other self-supervised or unsupervised anomaly detection algorithms to be used in place of the chosen DINO and FastFlow models.

### SPOTS Dataset

The full SPOTS dataset is closed to ensure the privacy of the game reserve and the animals under its protection, and to avoid possible nefarious use of the data.
While the full SPOTS dataset must remain private, a subset has been included in the _data_ folder in order to demonstrate the full functioning of both pipelines.

### Usage

- The example notebooks WILD_Self_Supervised_Pipeline.ipynb and WILD_Unsupervised_Pipeline.ipynb demonstrate the creation of labels from self-supervised attention maps and unsupervised anomaly detection segmentation masks respectively.
- The notebooks are set up for easy use in a Google Colab environment.
- To label your data, upload the _wild_ folder to Google Drive, upload your images to the relevant folders and use the tutorial notebooks as a guide to each pipeline.
