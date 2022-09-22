#The function of this module is to construct the optimal model for each month and save it#

import BaseFunc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data                                                
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time
mon_inEng={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'}

var='pr'# Or var='tas'
##Build DNN model
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        #Define Hidden Layer1
        self.hidden1=nn.Linear(in_features=16,out_features=1024,bias=True)#16 inpute GCMs

        self.hidden2=nn.Linear(1024,512)

        self.hidden3=nn.Linear(512,256)
    
        self.hidden4=nn.Linear(256,128)
       
        #Regression Prediction Output Layer
        self.predict=nn.Linear(128,1)  
    def forward(self,x):
        x=F.relu(self.hidden1(x))
     
        x=F.relu(self.hidden2(x))
     
        x=F.relu(self.hidden3(x))
 
        x=F.relu(self.hidden4(x))
 
        output=self.predict(x)
        return output[:,0]
if __name__ == '__main__':

 obs_img_path=r'F:\TestDemo1\obs_pr__1967_2.tif'   
 true_pixel_index=BaseFunc.GetTPixelIndex(obs_img_path)
 for mon in mon_inEng:
    train_month_path=r'F:\5_TrainingDataSet'+"\\"+str(var)+"\\"+mon+"\\TrainNet"
    train_year_folders_list=BaseFunc.GetSubfoldOfMonth(train_month_path)

    pred_month_path=r'F:\5_TrainingDataSet'+"\\"+str(var)+"\\"+mon+"\\TestNet"
    pred_year_folders_list=BaseFunc.GetSubfoldOfMonth(pred_month_path)

    x_train=[]
    y_train=[]
    x_pred=[]
    y_pred=[]
    #Get training and 
    BaseFunc.GetXYDataset(train_year_folders_list,x_train,y_train,true_pixel_index)
    BaseFunc.GetXYDataset(pred_year_folders_list,x_pred,y_pred,true_pixel_index)
    X_train=np.array(x_train)
    X_test=np.array(x_pred)
    y_train=np.array(y_train)
    y_test=np.array(y_pred)
    
    #standardized processing
    scale=StandardScaler()
    X_train_s=scale.fit_transform(X_train)
    X_test_s=scale.fit_transform(X_test)


    #Convert the dataset to a tensor and process it into data used by the DNN
    train_xt=torch.from_numpy(X_train_s.astype(np.float32))
    train_xt=train_xt.cuda()
    torch.save(train_xt, r'F:\6_1_MPLSort\Tensor'+"\\"+str(var)+"\\"+mon+"\\"+"pr_"+mon+"_train_xt.pt")

    train_yt=torch.from_numpy(y_train.astype(np.float32))
    train_yt=train_yt.cuda()
    torch.save(train_yt, r'F:\6_1_MPLSort\Tensor'+"\\"+str(var)+"\\"+mon+"\\"+"pr_"+mon+"_train_yt.pt")

    test_xt=torch.from_numpy(X_test_s.astype(np.float32))
    torch.save(test_xt, r'F:\6_1_MPLSort\Tensor'+"\\"+str(var)+"\\"+mon+"\\"+"pr_"+mon+"_test_xt.pt")

    test_yt=torch.from_numpy(y_test.astype(np.float32))
    torch.save(test_yt, r'F:\6_1_MPLSort\Tensor'+"\\"+str(var)+"\\"+mon+"\\"+"pr_"+mon+"_test_yt.pt")

    #Process data as data loader
    train_data=Data.TensorDataset(train_xt,train_yt)
    test_data=Data.TensorDataset(test_xt,test_yt)
    train_loader=Data.DataLoader(dataset=train_data,batch_size=1024 ,shuffle=True,num_workers=0)



    #Specifies that the model runs on the GPU
    device="cuda" if torch.cuda.is_available() else "cpu"
    mlpreg=MLPregression().to(device)
    print(mlpreg)


    #Define the optimizer
    optimizer=torch.optim.SGD(mlpreg.parameters(),lr=0.001,weight_decay=0.0001)
    loss_func=nn.MSELoss()

    train_loss_all=[]
    for epoch in range(2000):
        time_start=time.time()
        
        train_loss=0
        train_num=0
        #with torch.no_grad():
        for step,(b_x,b_y) in enumerate(train_loader):
            output=mlpreg(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*b_x.size(0)
            train_num+=b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        time_end=time.time()
        print(epoch,'--','time cost',time_end-time_start,'s')
        print(str(epoch)+"-------"+str(train_loss))
        torch.save(mlpreg, r'F:\6_1_MPLSort\Model'+"\\"+str(var)+"\\"+mon+"\\"+"pr_"+mon+"_module_"+str(epoch)+".pt") 
    print(mon)

    print("----success")
  

