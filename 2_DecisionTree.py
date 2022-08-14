from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math
from sklearn import metrics
import BaseFunc
mon_inEng={'Jan','Feb'}
if __name__ == '__main__':
    
    obs_img_path=r'F:\TestDemo1\obs_pr__1967_2.tif'
    true_pixel_index = BaseFunc.GetTPixelIndex(obs_img_path)
    
    R_accuracy_list=list(range(12))
    R2_accuracy_list=list(range(12))
    std_accuracy_list=list(range(12))
    CRMSE_accuracy_list=list(range(12))
    MAE_accuracy_list=list(range(12))

    for mon in mon_inEng:
        print("month----"+str(mon))
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
        clf = tree.DecisionTreeRegressor()
        clf.fit(x_train, y_train)
        
    
        # Plot outputs
        y_pre_test = clf.predict(x_test)
     
        #输出训练后模型y值和真实值y之间误差
        #输出相关性
        R=np.corrcoef(np.array(y_test), np.array(y_pre_test))[0][1]#np自带函数计算
        R_accuracy_list[mon_num-1]=R
        # #输出R2拟合度
        # R2=r2_score(y_test, y_pre_test)
        # R2_accuracy_list[mon_num-1]=R2
        # #计算std方差
        # std_accuracy=math.sqrt(np.var(np.array(y_pre_test)));print(std_accuracy)
        # std_accuracy_list[mon_num-1]=std_accuracy
        # #计算中心化均方根误差
        # # y_test[:]=y_test[:]-sum(y_test)/len(y_test);y_pre_test[:]=y_pre_test[:]-sum(y_pre_test)/len(y_pre_test)
        rmes=mean_squared_error(y_test, y_pre_test,squared=False)
        # # CRMSE_accuracy_list[mon_num-1]=rmes
        # #计算MAE
        MAE=metrics.mean_absolute_error(y_pre_test, y_test)
        MAE_accuracy_list[mon_num-1]=MAE
    for MAE in R_accuracy_list:
        print(MAE)
    

        


