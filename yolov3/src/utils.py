import torch 
import torch.nn as nn 
import torch.optim as optim 
  
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
  
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 
  
import os 
import numpy as np 
import pandas as pd 
  
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
  
from tqdm import tqdm

import config

# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes): 
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(config.class_labels))] 
  
    # Reading the image with OpenCV 
    img = np.array(image) 
    # Getting the height and width of the image 
    h, w, _ = img.shape 
  
    # Create figure and axes 
    fig, ax = plt.subplots(1) 
  
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes and labels over the image 
    count = 0
    for box in boxes: 
        # Get the class from the box 
        class_pred = box[0] 
        # Get the center x and y coordinates 
        box = box[2:] 
        # Get the upper left corner coordinates 
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
  
        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            box[2] * w, 
            box[3] * h, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
          
        # Add the patch to the Axes 
        ax.add_patch(rect) 
          
        # Add class name to the patch 
        plt.text( 
            upper_left_x * w, 
            upper_left_y * h, 
            s=config.class_labels[int(class_pred)], 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 
        count += 1
    # Display the plot 
        plt.savefig('/Users/pavithrak/CUAD/Pavithra_Repo/yolov3/src/mast_data/train_data/' + 'train_{}.jpg'.format(count))
    plt.show()