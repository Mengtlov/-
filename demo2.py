# -*- coding: utf-8 -*-
#类内单样本增量学习实验
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
    
    
trainX      = data[b'x_train_new']
trainY      = data[b'trainY']
testX       = data[b'x_test_new']
testY       = data[b'testY']
del data

#用2个类进行训练
#训练集，用于构造分类器
index=np.nonzero(trainY<=1)
trainX = trainX[index]
trainY = trainY[index]
#测试集，用于测试分类器
index=np.nonzero(testY<=1)
testX = testX[index]
testY = testY[index]

index=np.nonzero(testY==0)
testX0 = testX[index]
testY0 = testY[index]

index=np.nonzero(testY==1)
testX1 = testX[index]
testY1 = testY[index]


pdf_in1=[]
acc_in1=[]
acc0=[]
acc1=[]
for i in range(10,len(trainY)):
    #训练
    Y=trainY[:i]
    X=trainX[:i]
    label = np.unique(Y)
    nc = label.shape[0]
    nf = X.shape[1]
    p_y=PDL(Y)
    pdfs=PDF_x_Y(nc=nc,nf=nf,X=X,Y=Y)
    # 样本个数532 下标0-531 i+2为100的倍数，记录pdf曲线
    if (i+69)%100==0:
        pdf_in1.append(pdfs[0][0])
    #测试
    pre = Classifier(testX,p_y,pdfs,label=label,nf=nf,nc=nc)
    acc=np.count_nonzero(pre==testY)
    acc=acc*1.0/len(testY)
    acc_in1.append(acc)
    
    pre = Classifier(testX0,p_y,pdfs,label=label,nf=nf,nc=nc)
    acc=np.count_nonzero(pre==testY0)
    acc=acc*1.0/len(testY0)
    acc0.append(acc)
    
    pre = Classifier(testX1,p_y,pdfs,label=label,nf=nf,nc=nc)
    acc=np.count_nonzero(pre==testY1)
    acc=acc*1.0/len(testY1)
    acc1.append(acc)
#acc
plt.figure(figsize=(5.2,3.9))
plt.plot(acc_in1)
plt.xlabel('样本个数')
plt.ylabel('二分类准确率')
plt.legend()
plt.show()

#acc
plt.figure(figsize=(5.2,3.9))
plt.plot(acc0)
plt.xlabel('样本个数')
plt.ylabel('类别0识别准确率')
plt.legend()
plt.show()

#acc
plt.figure(figsize=(5.2,3.9))
plt.plot(acc1)
plt.xlabel('样本个数')
plt.ylabel('类别1识别准确率')
plt.legend()
plt.show()


#画概率密度函数
plt.figure(figsize=(5.2,3.9))
n=31.0
for i in pdf_in1:
    plt.plot(i[0],i[1],label=str(round(n/5.32))+'%的样本')
    n=n+100
plt.xlabel('特征值')
plt.ylabel('标签为0的条件下特征x0的概率密度')
plt.legend()
plt.show()

with open('完全训练所有条件pdf.pickle', 'rb') as handle:
    exp1pdfs = pickle.load(handle)

plt.figure(figsize=(5.2,3.9))   
plt.plot(pdf_in1[-1][0],pdf_in1[-1][1],label='100%的样本')
plt.plot(exp1pdfs[0][0][0],exp1pdfs[0][0][1],label='第一次试验')
plt.xlabel('特征值')
plt.ylabel('标签为0的条件下特征x0的概率密度')
plt.legend()
plt.show()
#for i in pdf_in1:
#    plt.plot(i[0],i[1])
    
with open('单样本增量正确率变化.pickle', 'wb') as handle:
    pickle.dump(acc_in1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('单样本增量pdf变化100一点.pickle', 'wb') as handle:
    pickle.dump(pdf_in1, handle, protocol=pickle.HIGHEST_PROTOCOL)    