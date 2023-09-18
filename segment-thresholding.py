import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob

import IPython.display as ipd
from tqdm import tqdm

import subprocess

plt.style.use('ggplot')

input_file = '/kaggle/input/segment-image-data/video.mp4'
subprocess.run(['ffmpeg',
                '-i',
                input_file,
                '-qscale',
                '0',
                'output.mp4',
                '-loglevel',
                'quiet']
              )
			  
!ls -GFlash --color

ipd.Video('output.mp4', width=700)

# Load in video capture
cap = cv2.VideoCapture('output.mp4')

# Total number of frames in video
cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Video height and width
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f'Height {height}, Width {width}')

# Get frames per second
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'FPS : {fps:0.2f}')

cap.release()

cap = cv2.VideoCapture('output.mp4')
ret, img = cap.read()
print(f'Returned {ret} and img of shape {img.shape}')

## Helper function for plotting opencv images in notebook
def display_cv2_img(img, figsize=(10, 10)):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_)
    ax.axis("off")
	
display_cv2_img(img)

cap.release()

fig, axs = plt.subplots(5, 5, figsize=(30, 20))
axs = axs.flatten()

cap = cv2.VideoCapture("output.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 100 == 0:
        axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[img_idx].set_title(f'Frame: {frame}')
        axs[img_idx].axis('off')
        img_idx += 1

plt.tight_layout()
plt.show()
cap.release()

# Pull frame 1035

cap = cv2.VideoCapture("output.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame == 1035:
        break
cap.release()

display_cv2_img(img)

img_example = img.copy()
#frame_labels = video_labels.query('video_frame == 1035')
#for i, d in frame_labels.iterrows():
#pt1 = int(d['box2d.x1']), int(d['box2d.y1'])
#pt2 = int(d['box2d.x2']), int(d['box2d.y2'])
#pt1 = int('box2d.x1'), int('box2d.y1')
#pt2 = int('box2d.x2'), int('box2d.y2')
#cv2.rectangle(img_example, pt1, pt2, (0, 0, 255), 3)
#cv2.rectangle(img_example, (0, 0, 255), 3)

display_cv2_img(img_example)


!pip install scikit-image

# Importing Necessary Libraries
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
data = img_example
plt.subplot(1, 2, 1)

# Displaying the sample image
plt.imshow(data)

# Converting RGB image to Monochrome
gray_data = rgb2gray(data)
plt.subplot(1, 2, 2)

# Displaying the sample image - Monochrome
# Format
plt.imshow(gray_data, cmap="gray")



# Importing necessary libraries
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
data = img_example
gray_data = rgb2gray(data)

# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_data)

# Computing binarized values using the obtained
# threshold
binarized_data = (gray_data > threshold)*1
plt.subplot(2,2,1)
plt.title("Threshold: >"+str(threshold))

# Displaying the binarized image
plt.imshow(binarized_data, cmap = "gray")

# Computing Ni black's local pixel
# threshold values for every pixel
threshold = filters.threshold_niblack(gray_data)

# Computing binarized values using the obtained
# threshold
binarized_data = (gray_data > threshold)*1
plt.subplot(2,2,2)
plt.title("Niblack Thresholding")

# Displaying the binarized image
plt.imshow(binarized_data, cmap = "gray")

# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_data)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")

# Displaying the local threshold values
plt.imshow(threshold, cmap = "gray")

# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_data = (gray_data > threshold)*1
plt.subplot(2,2,4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")

# Displaying the binarized image
plt.imshow(binarized_data, cmap = "gray")


