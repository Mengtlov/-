# -*- coding: utf-8 -*-
#完全训练实验
from BayesClassifier import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats

font = {'family' : 'NSimSun',   
        'weight' : 'normal',  
        'size'   : 10.05,  
        }  
plt.rc('font',**font) 

with open('predict_train_test_data.pickle', 'rb') as handle:
    data = pickle.load(handle,encoding='bytes')
    
    
trainX      = data[b'x_train_new']
trainY      = data[b'trainY']
testX       = data[b'x_test_new']
testY       = data[b'testY']
del data

#用所有类进行训练

#训练 学习p(y)和p(x|y)
label = np.unique(trainY)
nc = label.shape[0]
nf = trainX.shape[1]
p_y=PDL(trainY)
pdfs=PDF_x_Y(nc=nc,nf=nf,X=trainX,Y=trainY)


with open('完全训练所有条件pdf.pickle', 'wb') as handle:
    pickle.dump(pdfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
#测试 对训练集和测试集分别进行
#训练集
pre = Classifier(trainX,p_y,pdfs,label=label,nf=nf,nc=nc)
acc=np.count_nonzero(pre==trainY)
acc=acc*1.0/len(trainY)
print(acc*100,'%') 

#测试集
pre = Classifier(testX,p_y,pdfs,label=label,nf=nf,nc=nc)
acc=np.count_nonzero(pre==testY)
acc=acc*1.0/len(testY)
print(acc*100,'%')
 
#评价
#target_names = ['class 0(2S1)', 'class 1(BMP2)', 'class 2(BRDM2)',
#                'class 3(BTR70)', 'class 4(BTR60)', 'class 5(D7)',
#                'class 6(T62)', 'class 7(T72)', 'class 8(ZIL131)',
#                'class 9(ZSU234)']

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
target_names = ['2S1', 'BMP2', 'BRDM2','BTR70', 'BTR60', 'D7','T62', 'T72', 'ZIL131','ZSU234']
print(classification_report(testY,
                            pre,
                            target_names=target_names))
print(confusion_matrix(testY,
                       pre))
#画概率密度函数
plt.figure(figsize=(5.2,3.9))
plt.plot(pdfs[0][0][0],pdfs[0][0][1],label='p(x0|y=0)')#[类][特征][x,y]
plt.xlabel('特征值')
plt.ylabel('概率密度')
plt.legend()
plt.show()
