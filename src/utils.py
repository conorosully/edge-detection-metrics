#A list of functions that are used in the main script

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from osgeo import gdal
from skimage import feature

import os
import glob
import random

from scipy.ndimage import distance_transform_edt

#gloabl variables
global channels 
channels = ['Coastal Aerosol','Blue','Green',
                 'Red','Red Edge 1','Red Edge 2',
                'Red Edge 3','NIR','Red Edge 4',
                 'Water Vapour','SWIR 1','SWIR 2']

# Functions
def get_rgb(img):
    """Return normalized RGB channels from sentinal image"""
    
    rgb_img = img[:, :, [3,2,1]]
    rgb_normalize = np.clip(rgb_img/10000, 0, 0.3)/0.3
    
    return rgb_normalize

def load_test(path):
    """Returns sentinal image, rgb image and label"""
    
    img = gdal.Open(path).ReadAsArray()
    stack_img = np.stack(img, axis=-1)
    rgb_img = get_rgb(stack_img)
    
    label_path = path.replace("images","labels").replace("image","label")
    label = gdal.Open(label_path).ReadAsArray()
    
    return stack_img, rgb_img, label

def fom(ref_img,img, alpha = 1.0 / 9):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """
    
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(ref_img))

    fom = 1.0 / np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(ref_img))

    N, M = img.shape

    for i in range(N):
        for j in range(M):
            if img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(ref_img))    

    return fom

def preprocess(img_input):
        """Preprocess image for edge detection"""
        img = img_input.copy()
        img = np.array(img)

        # Iterate over bands
        for i in range(12):
                img_i = img[:,:,i]

                # Scale bands between 0 and 255
                img[:,:,i] = cv2.normalize(img[:,:,i], None, 0, 255, cv2.NORM_MINMAX)
                img_i = np.uint8(img_i)
 
                img[:,:,i] = img_i
       
        img = np.uint8(img)
        return img


def canny_ed(img_input, threshold1=100, threshold2=200):

    """Apply canny edge detection to image"""
    
    img = img_input.copy()
    img = preprocess(img)

    # Iterate over bands
    for i in range(12):
        img_i = img[:,:,i]

        #kernel = (5, 5)
        #img_i = cv2.GaussianBlur(img_i, kernel,0)

        #img_i = cv2.normalize(img_i, None, 0, 255, cv2.NORM_MINMAX)
        
        img_i = cv2.Canny(img_i,threshold1 = threshold1, threshold2 = threshold2)
        #img_i = feature.canny(img_i, sigma=2, low_threshold=threshold1, high_threshold=threshold2)


        img[:,:,i] = img_i

    return img

def plot_metric_trends(df,metric):

    """Plot trends of metrics for different thresholds and bands"""
    
    thresholds = df['thresholds'].unique()

    mean =  df.groupby(["thresholds","band"],as_index=False).mean()
    sd =  df.groupby(["thresholds","band"],as_index=False).std()

    fig,ax = plt.subplots(1,1,figsize=(15,5))

    w = 0.4
    for i,thresh in enumerate(thresholds):

        bands = mean[mean.thresholds==str(thresh)]["band"]

        mean_i = mean[mean.thresholds==str(thresh)][metric]
        sd_i = sd[sd.thresholds==str(thresh)][metric]
        
        plt.bar(bands-w/3+i*w/3,mean_i,yerr=sd_i,width=w/3,label=str(thresh))

    plt.ylabel(metric.upper(),fontsize=20)
    plt.xticks(ticks=range(1,13),labels= channels,fontsize=15, rotation=90)

    plt.legend(title="Thresholds")