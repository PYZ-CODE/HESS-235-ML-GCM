import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data

import os
from osgeo import gdal,osr,ogr
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def GetMonthNum(mon):
 if mon=='Jan':
     mon_num=1
 elif mon=='Feb':
     mon_num=2
 elif mon=='Mar':
     mon_num=3
 elif mon=='Apr':
     mon_num=4
 elif mon=='May':
     mon_num=5
 elif mon=='Jun':
     mon_num=6
 elif mon=='Jul':
     mon_num=7
 elif mon=='Aug':
     mon_num=8
 elif mon=='Sept':
     mon_num=9
 elif mon=='Oct':
     mon_num=10
 elif mon=='Nov':
     mon_num=11
 elif mon=='Dec':
     mon_num=12
 return mon_num

#Get the row and column number of the pixel with real value in Observation dataset
def GetTPixelIndex(obs_img_path):
    obs_img_path = str(obs_img_path)
    dataset = gdal.Open(obs_img_path)
    print(dataset.GetDescription())#data description
    cols = dataset.RasterXSize#Number of image columns
    rows = (dataset.RasterYSize)#image lines

    im_data_ori = dataset.ReadAsArray(0, 0, cols, rows)
    #true_index = ~np.isnan(im_data_ori) #Return TrueFalse matrix
    true_index=np.argwhere(~np.isnan(im_data_ori))
    del dataset
    return true_index

#Get the subfolder directory in the month file (the subfolder is based on the year)
def GetSubfoldOfMonth(first_in_path):
    year_folder_list = []
    for home, dirs, files in os.walk(first_in_path): 
        for dir in dirs:
            if(str.isdigit(dir)):
             year_folder_list.append(os.path.join(home, dir))
    return year_folder_list 

#Get the paths of all tifs in subfolders, in list form
def GetYearTIFPath(year_folder):
  year_tif_list=[] 
  for p in Path(year_folder).rglob('*.tif'): 
    # yield s
    test_iter=Path(year_folder).iterdir()
    year_tif_list.append(p)
  year_tif_list.sort()
  return year_tif_list

#Construct the true value array for each year of the CRU (one-dimensional column vector)
def GetPerYearObs(tif,if_matrix):
 obs_var_vector=[]
 tif_path=str(tif)   
 dataset = gdal.Open(tif_path)
 var_matrix = dataset.ReadAsArray(0, 0, 720, 360)

 for eff_index in if_matrix:
      eff_pixel_var=var_matrix[eff_index[0]][eff_index[1]]
      obs_var_vector.append(eff_pixel_var)
 return obs_var_vector

#Insert the true value array for each year (one-dimensional column vector)
def GetPerYearGCM(tif,if_matrix,perpixel_GCMs):
 tif_path=str(tif)   
 dataset = gdal.Open(tif_path)
 var_matrix = dataset.ReadAsArray(0, 0, 720, 360)
 for i in range(len(if_matrix)):
     row_index=if_matrix[i][0]
     col_index=if_matrix[i][1]
     pixel_var =var_matrix[row_index][col_index] 
     perpixel_GCMs[i].append(pixel_var) 

#Read XY training or prediction values ​​for each year
def GetXYDataset(year_folders_list,x_total,y_total,true_pixel_index):
 for y_f_l in year_folders_list:
    tif_list=GetYearTIFPath(y_f_l)# Get 16 patterns plus 1 true value TIF path for each year
    peryear_x = [[] for i in range (len(true_pixel_index)) ]# Get the 16 pattern set values ​​of all pixels for each year
    for tif in tif_list:
     
     tif_pathjname=str(tif)
     if (tif_pathjname.split('\\')[-1].split('.')[0].split('_')[0]=='obs'):
      peryear_y = GetPerYearObs(tif,true_pixel_index)
      y_total.extend(peryear_y)
     else:#if GCMs 
      GetPerYearGCM(tif,true_pixel_index,peryear_x)#Each time it is executed, a pattern [year1_pixel1[Model1,Model2,...,Model16],year1_pixel2[Model1,Model2,...,Model16]] is populated
      
    x_total.extend(peryear_x) 

def WriteSSPTIF(true_pixel_index,y_predict,TIF_path):
    driver = gdal.GetDriverByName('GTiff')
    out_tif_row=360;out_tif_col=720
    out_tif = driver.Create(TIF_path,out_tif_col,out_tif_row,1,gdal.GDT_Float32)
    

    LonMin,LatMax,LonMax,LatMin = [0,90,360,-90]
    geotransform = (LonMin,0.5, 0, LatMax, 0, -0.5)
    out_tif.SetGeoTransform(geotransform)
    
    img_data = np.full([360,720], np.nan)
#fill raster data matrix
    for index,y in zip(true_pixel_index,y_predict):
        img_data[index[0]][index[1]]=y
    a=img_data[12][284]
    b=img_data[12][435]
 

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) # Define the output coordinate system as "WGS 84", AUTHORITY["EPSG","4326"]
    out_tif.SetProjection(srs.ExportToWkt()) # Assign projection information to the new layer
    out_tif.GetRasterBand(1).WriteArray(img_data)
    out_tif.FlushCache() # # write data to disk
    print(TIF_path)

