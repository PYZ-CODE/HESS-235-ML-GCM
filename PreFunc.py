from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data

def GetAllModule(module_path):
  module_list=[] 
  for p in Path(module_path).rglob('*.pt'): 
    module_list.append(p)
  return module_list

def CalAccuracy(module_train_path,test_xt,test_yt):
    module_cpu=torch.load(module_train_path).cpu()
    y_pre_test=module_cpu(test_xt).tolist()
    test_yt=test_yt.tolist()
    test_R2=r2_score(test_yt, y_pre_test)
 #Output the correlation between the model y value after training and the true value y
    R=np.corrcoef(np.array(test_yt), np.array(y_pre_test))[0][1]
    print("R-----"+str(R))
  
    R2=r2_score(test_yt, y_pre_test)
    print("R2-----"+str(R2))
    
    std_accuracy=math.sqrt(np.var(np.array(y_pre_test)))
    print("std-----"+str(std_accuracy))
  
    test_yt[:]=(np.array(test_yt)-sum(np.array(test_yt))/len(np.array(test_yt)));y_pre_test[:]=np.array(y_pre_test)-sum(np.array(y_pre_test))/len(np.array(y_pre_test))
    rmes=mean_squared_error(test_yt, y_pre_test,squared=False) 
    print("rmes-----"+str(rmes))
 
    fenzi=np.array(test_yt)-np.array(y_pre_test)
    a=sum(np.absolute(fenzi))
    b=len(test_yt)
    MAE=sum(np.absolute(fenzi))/len(test_yt)
    print("MAE-----"+str(MAE))

# Traverse different models in the same month to select the best
def CalAccuracy_2(module_train_path,test_xt,test_yt,path_list,R_list,R2_list,std_list,rems_list,MAE_list):
    module_cpu=torch.load(module_train_path).cpu()
    
    epoch=int(str(module_train_path).split('\\')[-1].split('.')[0].split('_')[-1])

    path_list[epoch]=module_train_path
    y_pre_test=module_cpu(test_xt).tolist()
    test_yt=test_yt.tolist()
    test_R2=r2_score(test_yt, y_pre_test)
    # Traverse different models in the same month to select the best
    R=np.corrcoef(np.array(test_yt), np.array(y_pre_test))[0][1]
    print("R-----"+str(R))
    R_list[epoch]=R
   
    R2=r2_score(test_yt, y_pre_test)
    print("R2-----"+str(R2))
    R2_list[epoch]=R2

   
    std_accuracy=math.sqrt(np.var(np.array(y_pre_test)))
    print("std-----"+str(std_accuracy))
    std_list[epoch]=std_accuracy

 
    test_yt[:]=(np.array(test_yt)-sum(np.array(test_yt))/len(np.array(test_yt)));y_pre_test[:]=np.array(y_pre_test)-sum(np.array(y_pre_test))/len(np.array(y_pre_test))
    rmes=mean_squared_error(test_yt, y_pre_test,squared=False) 
    print("rmes-----"+str(rmes))
    rems_list[epoch]=rmes

 
    fenzi=np.array(test_yt)-np.array(y_pre_test)
    a=sum(np.absolute(fenzi))
    b=len(test_yt)
    MAE=sum(np.absolute(fenzi))/len(test_yt)
    print("MAE-----"+str(MAE))
    MAE_list[epoch]=MAE                    



