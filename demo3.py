# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:00:59 2018

@author: Administrator
"""

#新类增量学习实验
from BayesClassifier import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats

font = {'family' : 'NSimSun,Times New Roman',   
        'weight' : 'normal',  
        'size'   : 10.05,  
        }  
plt.rc('font',**font) 

with open('predict_train_test_data.pickle', 'rb') as handle:
    data = pickle.load(handle,encoding='bytes')
    
    
trainX_all      = data[b'x_train_new']
trainY_all      = data[b'trainY']
testX_all       = data[b'x_test_new']
testY_all       = data[b'testY']
del data

#最初两类的测试集
index=np.nonzero(testY_all<=1)
testX2 = testX_all[index]
testY2 = testY_all[index]
#标签0
index=np.nonzero(testY_all==0)
testX0 = testX_all[index]
testY0 = testY_all[index]
#标签1
index=np.nonzero(testY_all==1)
testX1 = testX_all[index]
testY1 = testY_all[index]
#训练集，用于构造分类器
pdf_new_class=[]
acc_old_2class=[]
acc_new_class=[]
acc_class0=[]
for l in range(1,10):
    #每次训练增加一个类
    index=np.nonzero(trainY_all<=l)
    trainX = trainX_all[index]
    trainY = trainY_all[index]
    #测试集，用于测试分类器
    index=np.nonzero(testY_all<=l)
    testX = testX_all[index]
    testY = testY_all[index]
    
    #训练
    label = np.unique(trainY)
    nc = label.shape[0]
    nf = trainX.shape[1]
    
    p_y=PDL(trainY)
    pdfs=PDF_x_Y(nc=nc,nf=nf,X=trainX,Y=trainY)
    # 每学习完一个新类记录pdf曲线
    pdf_new_class.append(pdfs[0][0])
    
    #测试所有已学类
    pre = Classifier(testX,p_y,pdfs,label=label,nf=nf,nc=nc)
    acc=np.count_nonzero(pre==testY)
    acc=acc*1.0/len(testY)
    acc_new_class.append(acc)
    
    #测试最初两类类
    pre = Classifier(testX0,p_y,pdfs,label=label,nf=nf,nc=nc)
    acc=np.count_nonzero(pre==testY0)
    acc=acc*1.0/len(testY0)
    acc_class0.append(acc)

with open('新类增量0类0特征pdf变化.pickle', 'wb') as handle:
    pickle.dump(pdf_new_class, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('新类增量测试准确率变化.pickle', 'wb') as handle:
    pickle.dump(acc_new_class, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('新类增量对原始2类测试准去率变化.pickle', 'wb') as handle:
    pickle.dump(acc_old_2class, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#for i in pdf_new_class:
#    plt.plot(i[0],i[1])
#    plt.show()
plt.figure(figsize=(5.2,3.9)) 
m=2
for i in pdf_new_class:
    plt.plot(i[0],i[1],label='增量到'+str(m)+'类')
    m=m+1
plt.xlabel('特征值')
plt.ylabel('标签为0的条件下特征x0的概率密度')
plt.legend()
plt.show()


plt.figure(figsize=(5.2,3.9)) 

plt.plot(acc_class0,label='对类别0的预测准确率')

plt.xlabel('类别数')
plt.ylabel('准确率')
plt.legend()
plt.show()