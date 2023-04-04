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


class data_processor: 

    def __init__(self):
         pass

    def get_rgb(self,img):
        """Return normalized RGB channels from sentinal image"""
        
        rgb_img = img[:, :, [3,2,1]]
        rgb_normalize = np.clip(rgb_img/10000, 0, 0.3)/0.3
        
        return rgb_normalize

    def load_test(self,path):
        """Returns sentinal image, rgb image and label"""
        
        img = gdal.Open(path).ReadAsArray()
        stack_img = np.stack(img, axis=-1)
        rgb_img = self.get_rgb(stack_img)
        
        label_path = path.replace("images","labels").replace("image","label")
        label = gdal.Open(label_path).ReadAsArray()
        
        return stack_img, rgb_img, label
    
    def fom(self,ref_img,img, alpha = 1.0 / 9):
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

    def get_rates(self,ref_img,img):
            """Calculate true positive, false positive, true negative and false negative rates"""

            h,w = img.shape

            p = np.count_nonzero(ref_img)
            n = np.count_nonzero(np.logical_not(ref_img))

            
            tp = np.count_nonzero(np.logical_and(img,ref_img))
            fp = np.count_nonzero(np.logical_and(img,np.logical_not(ref_img)))
            tn = np.count_nonzero(np.logical_and(np.logical_not(img),np.logical_not(ref_img)))
            fn = np.count_nonzero(np.logical_and(np.logical_not(img),ref_img))

            tpr = tp/p
            fpr = fp/n
            tnr = tn/n
            fnr = fn/p

            fp_fn_ratio = fp/fn

            return tp,fp,tn,fn,tpr,fpr,tnr,fnr,fp_fn_ratio

    def preprocess(self,img_input):
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


    def canny_ed(self,img_input, threshold1=100, threshold2=200):

        """Apply canny edge detection to image"""
        
        img = img_input.copy()
        img = self.preprocess(img)

        # Iterate over bands
        for i in range(12):

            img_i = img[:,:,i]
            
            img_i = cv2.Canny(img_i,threshold1 = threshold1, threshold2 = threshold2)
            img[:,:,i] = img_i

        return img
     

class data_visualizer:

    def __init__(self,df_metrics,canny,edge_reference,rgb,thresholds,channels):
        self.df_metrics = df_metrics
        self.canny = canny
        self.edge_reference = edge_reference
        self.rgb = rgb
        self.thresholds = thresholds
        self.channels = channels

    def plot_metric_trends(self,metric,ylabel=None,save_path=None):

        """Plot trends of metrics for different thresholds and bands"""
        
        thresholds = self.df_metrics['thresholds'].unique()

        mean =  self.df_metrics.groupby(["thresholds","band"],as_index=False).mean()
        sd =  self.df_metrics.groupby(["thresholds","band"],as_index=False).std()

        fig,ax = plt.subplots(1,1,figsize=(15,5))

        w = 0.4
        for i,thresh in enumerate(thresholds):

            bands = mean[mean.thresholds==str(thresh)]["band"]

            mean_i = mean[mean.thresholds==str(thresh)][metric]
            sd_i = sd[sd.thresholds==str(thresh)][metric]
            
            plt.bar(bands-w/3+i*w/3,mean_i,yerr=sd_i,width=w/3,label=str(thresh))

        if ylabel:
             plt.ylabel(ylabel,fontsize=20)
        else:
            plt.ylabel(metric.upper(),fontsize=20)


        plt.xticks(ticks=range(1,13),labels= self.channels,fontsize=15, rotation=90)
        plt.yticks(fontsize=15)

        legend = plt.legend(title="Thresholds",fontsize=12,loc=(1.01, 0.44))
        legend.get_title().set_fontsize('15')

        if save_path:
            plt.tight_layout()
            fig.set_facecolor('white')
            plt.savefig(save_path)

    def plot_combined_metric_trends(self,metrics,ylabel=None,save_path=None):

        """Plot trends of metrics for different thresholds and bands"""
        
        thresholds = self.df_metrics['thresholds'].unique()

        mean =  self.df_metrics.groupby(["thresholds","band"],as_index=False).mean()
        sd =  self.df_metrics.groupby(["thresholds","band"],as_index=False).std()

        n = len(metrics)
        fig,ax = plt.subplots(n,1,figsize=(15,4*n))

        for j,metric in enumerate(metrics):
            
            w = 0.8
            for i,thresh in enumerate(thresholds):

                bands = mean[mean.thresholds==str(thresh)]["band"]

                mean_i = mean[mean.thresholds==str(thresh)][metric]
                sd_i = sd[sd.thresholds==str(thresh)][metric]
                
                ax[j].bar(bands-w/3+i*w/6,mean_i,yerr=sd_i,width=w/6,label=str(thresh))

            if ylabel:
                ax[j].set_ylabel(ylabel,fontsize=20)
            else:
                ax[j].set_ylabel(metric.upper(),fontsize=20)

            # Set xticks
            if j == n-1:
                ax[j].set_xticks(ticks=range(1,13),labels= self.channels,fontsize=20, rotation=90)
            else:
                ax[j].set_xticks(ticks=range(1,13),fontsize=15)
                ax[j].set_xticklabels([])
            
            # Set yticks
            #yticks = np.round(ax[j].get_yticks())
            #ax[j].set_yticklabels(yticks,fontsize=15)
            ax[j].tick_params(axis='y', which='major', labelsize=15)

            # Legend
            if j == 0:
                legend = ax[0].legend(title="Thresholds",fontsize=12,loc=(1.01, 0.35))
                legend.get_title().set_fontsize('15')

        plt.tight_layout(pad = 2)
        if save_path:
            fig.set_facecolor('white')
            plt.savefig(save_path)

    def get_title(self,ID,threshold,band = 7):
            """Get title for image. Set the best metric to bold"""
            
            IMG_ID = "IMG#{}".format(ID)
            df = self.df_metrics[(self.df_metrics['ID']==IMG_ID) & (self.df_metrics['band']==band)]
            df_thresh = df[df['thresholds']==str(threshold)]

            title = "\nRMSE: " 
            rmse_ = df_thresh['rmse'].values[0]

            if df['rmse'].min() == rmse_:
                    title += r"$\bf{" + str(round(rmse_,2)) + "}$" 
            else:
                    title += str(round(rmse_,2))
            
            for metric in ['psnr','ssim','fom']:
                    title += "\n" + metric.upper() + ": "
                    metric_ = df_thresh[metric].values[0]
                    if df[metric].max() == metric_:
                            title += r"$\bf{" + str(round(metric_,2)) + "}$"
                    else:
                            title +=  str(round(metric_,2))

            return title

    def example_plots(self,IDs,band=7,ex_diff = 0,show_metrics=True,save_path=None):
        """Plot example images and metrics"""

        fig, axs = plt.subplots(len(IDs), 7, figsize=(30, 5*len(IDs)+2))
        fig.set_facecolor('white')

        for i, ID in enumerate(IDs):

            axs[i,0].imshow(255-self.edge_reference[ID], cmap='gray')
            
            axs[i,0].set_ylabel("Example {}".format(i+1+ex_diff),size=25,weight='bold')

            if i == 0:
                axs[i,0].set_title("Thresholds:\n\n\n\n\nReference",size=20)
            else:
                axs[i,0].set_title("Ground Truth",size=20)



            for j, threshold in enumerate(self.thresholds):
                img = self.canny[str(threshold)][ID][:,:,band]

                axs[i,j+1].imshow(255-img, cmap='gray')

                title = self.get_title(ID, threshold)
                if i == 0:
                    title = str(threshold) + "\n" +  title
                
                if show_metrics==False:
                    title = str(threshold)
                
                axs[i,j+1].set_title(title,size=20)

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', dpi=300)


    def example_plots_rgb(self,IDs,band=7,save_path=None):
        """Plot example images and metrics"""

        fig, axs = plt.subplots(len(IDs), 8, figsize=(30, 5*len(IDs)+2))
        fig.set_facecolor('white')

        for i, ID in enumerate(IDs):

            axs[i,0].imshow(self.rgb[ID])
            axs[i,1].set_title("RGB",size=20)

            axs[i,1].imshow(255-self.edge_reference[ID], cmap='gray')
            axs[i,1].set_title("Reference",size=20)

            for j, threshold in enumerate(self.thresholds):
                img = self.canny[str(threshold)][ID][:,:,band]

                axs[i,j+2].imshow(255-img, cmap='gray')
                axs[i,j+2].set_title(self.get_title(ID, threshold),size=20)

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', dpi=300)