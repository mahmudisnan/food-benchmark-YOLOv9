'''
Author: Gaurav Singhal
'''

import json
import cv2
from pycocotools import coco, mask
import pycocotools.mask as maskUtils
import tqdm

# +
def read_annotation(ann_path: str):
    '''
    Read the annotations
    '''
    
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    return annotations

def reset_image_dims(images: list, image_path: str):
    '''
    Reset the image height and width in the annotations with original 
    image dimension
    '''
    for i in range(len(images)):
        dim_tup = cv2.imread(f"{image_path}/{images[i]['file_name']}").shape
        images[i]['height'], images[i]['width'] = dim_tup[0], dim_tup[1]
    return images

def preprocess_images(annotations: dict, image_path: str):
    '''
    1. Remove the bad images which have rotated annotations
    2. Reset the image height and width in the annotations with original 
    image dimension
    '''
    useless = []
    # Check for bad images
    for i in tqdm.tqdm(annotations['images']):
        im = cv2.imread(f"{image_path}/{i['file_name']}")
        if((im.shape[0]!=i['height']) or (im.shape[1]!=i['width'])):
            useless.append(i)
        
        # Update the dimensions
        i['height'], i['width'] = im.shape[0], im.shape[1]
        del im # Clean up memory

    # Remove bad images
    if len(useless) > 0:
        bad_ids = [item["id"] for item in useless]
        for i, item in enumerate(annotations['images']):
            if item["id"] in bad_ids:
                del annotations["images"][i]

        # Remove bad annotations for these images
        for i, item in enumerate(annotations['annotations']):
            if item["image_id"] in bad_ids:
                del annotations["annotations"][i]

    return annotations

def remove_bad_segmentation(segmentations: list):
    '''
    Remove segmentations which has less than 3 coordinates
    '''
    for i, ann in enumerate(tqdm.tqdm(segmentations)):
        segments = [seg for seg in ann['segmentation'] if len(seg)>=6]
        segmentations[i]['segmentation'] = segments
    return segmentations

def redraw_boxes(annotations: list, coco_ds):
    for item in tqdm.tqdm(annotations):
        try:
            # convert the item to a binary mask
            bin_mask = coco_ds.annToMask(item)
            new_bbox = mask.toBbox(mask.encode(bin_mask))
            item['bbox'] = list(new_bbox)

        except KeyError as e:
            print("Error with image", item['image_id'])
            print(type(e), e)
    return annotations

def preprocess_pipeline(work_dir: str):
    '''
    Executes the pre-processing pipeline
    '''
    # Read the annotations
    print("**Reading the annotations**")
    annotations = read_annotation(f"{work_dir}/annotations.json")
    coco_ds = coco.COCO(f"{work_dir}/annotations.json")
    print(f"Total images: {len(annotations['images'])},\
        \tTotal annotations: {len(annotations['annotations'])}")
    
    # Correct the images
    print("**Removing bad images and correcting image dimensions**")
    annotations = preprocess_images(annotations, f"{work_dir}/images")
    print(f"Total images: {len(annotations['images'])},\
        \tTotal annotations: {len(annotations['annotations'])}")
    
    # Remove bad segmentations
    print("**Removing segments with less than 3 coordinates**")
    annotations['annotations'] = remove_bad_segmentation(annotations['annotations'])
    print(f"Total images: {len(annotations['images'])},\
        \tTotal annotations: {len(annotations['annotations'])}")
    
    # Remove bad bounding boxes
    print("**Re-drawing bounding boxes**")
    annotations['annotations'] = redraw_boxes(annotations['annotations'], coco_ds)
    print(f"Total images: {len(annotations['images'])},\
        \tTotal annotations: {len(annotations['annotations'])}")
    
    return annotations
    
    
def do_preprocess():
    # For training annotations
    print("**Working with training annotations**")
    train_dir = "data/train"
    train_annotations = preprocess_pipeline(train_dir)
    save_annotations(train_annotations, train_dir)
    
    # For validation annotations
    print("\n\n**Working with validation annotations**")
    val_dir = "data/val"
    val_annotations = preprocess_pipeline(val_dir)
    save_annotations(val_annotations, val_dir)


def save_annotations(annotations: dict, work_dir: str):
    '''
    Save the annotations
    '''
    file_name = f"{work_dir}/annotations_new.json"
    with open(file_name, "w") as f:
        f.write(json.dumps(annotations))
    print(f"New annotations saved: {file_name}")


# -

do_preprocess() 



# +
# import matplotlib.pyplot as plt
# from pycocotools import coco
# import numpy as np
# ann = coco.COCO("data/train/annotations.json")
# def visualize(image_path: str):
#     # Get annotation
#     image_id = int(image_path.split("/")[-1].split(".")[0])
#     annIds = ann.getAnnIds(imgIds=[image_id])
#     anns = ann.loadAnns(annIds)
#     plt_im = plt.imread(image_path)
#     plt.imshow(plt_im)
#     #plt.imshow(np.rot90(plt_im, 3))
#     ann.showAnns(anns)
    
# visualize(f"{train_dir}/images/008934.jpg")
