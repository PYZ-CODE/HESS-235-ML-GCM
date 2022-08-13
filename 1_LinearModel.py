# 解析器选择 machinelearning
import os
from osgeo import gdal,osr,ogr
from sklearn import linear_model
import numpy as np
import glob
from pathlib import Path
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import BaseFunc
#mon_inEng={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'}
mon_inEng={'Apr','Feb'}

if __name__ == '__main__':
    

    obs_img_path=r'F:\TestDemo1\obs_pr__1967_2.tif'
    true_pixel_index = BaseFunc.GetTPixelIndex(obs_img_path)
    R_accuracy_list=list(range(12))
    R2_accuracy_list=list(range(12))
    std_accuracy_list=list(range(12))
    CRMSE_accuracy_list=list(range(12))
    MAE_accuracy_list=list(range(12))
    for mon in mon_inEng:
        mon_num=BaseFunc.GetMonthNum(mon)
        month_path_train=r'F:\5_TrainingDataSet\pr'+"\\"+mon+"\\TrainNet"
        year_folders_list_train=BaseFunc.GetSubfoldOfMonth(month_path_train)
        month_path_test=r'F:\5_TrainingDataSet\pr'+"\\"+mon+"\\TestNet"
        year_folders_list_test=BaseFunc.GetSubfoldOfMonth(month_path_test)
        
    
        x_train=[] ; y_train=[]#存储训练集或测试集中XY值
        BaseFunc.GetXYDataset(year_folders_list_train,x_train,y_train,true_pixel_index)
        x_test=[] ; y_test=[]#存储训练集或测试集中XY值
        BaseFunc.GetXYDataset(year_folders_list_test,x_test,y_test,true_pixel_index)
            
        #构建训练模型
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        result = reg.coef_

        # Plot outputs
        y_test_pre = reg.predict(x_test)
        y_test_pre=np.reshape(y_test_pre,())
      
        #计算MAE
        MAE=metrics.mean_absolute_error(y_pre_test, y_test)
        MAE_accuracy_list[mon_num-1]=MAE #预测值和真实值进行比较，评估误差
        

    for CRMSE in MAE_accuracy_list:
        print(CRMSE)

    a=0  
        

                  
                 


  