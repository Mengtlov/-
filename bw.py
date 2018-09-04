# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 19:15:33 2018

@author: Administrator
"""


import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats

font = {'family' : 'NSimSun,Times New Roman',   
        'weight' : 'normal',  
        'size'   : 10.05,  
        }  
plt.rc('font',**font) 



def PDL(Y):
    label = np.unique(Y)#列出所有标签
    #pdl = np.zeros_like(label)
    count = np.zeros_like(label)
    total = len(Y)
    for i in range(len(label)):#统计各个标签数量
        count[i] = np.count_nonzero(Y==label[i])
        
    pdl = count*1.0/total#计算各个类出现的概率
    
    return pdl

# probability density function 
def PDF(x):
    kde = stats.gaussian_kde(x, bw_method='scott')#使用stats中的高斯核非参数估计
    bw = kde.scotts_factor() * np.std(x)#带宽计算，使用
    print(bw)
    clip = (-np.inf, np.inf)
    gridsize = 100#pdf点数
    cut = 4
    support_min = max(x.min() - bw * cut, clip[0])
    support_max = min(x.max() + bw * cut, clip[1])
    grid = np.linspace(support_min, support_max, gridsize)
    y = kde(grid)
    x, y =grid, y
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)
    return [x,y]

def PDF_x_Y(nc,nf,X,Y):#nc类别数nf特征数
    nf = X.shape[1]
    label = np.unique(Y)
    nc = label.shape[0]
    pdfs=[]
    for i in range(nc):#pdf每个y有一组pdf，对应于各个特征p(xi|y),i=0,1,2...
        pdfs.append([])
        for j in range(nf):
            pdfs[i].append(PDF(X[np.nonzero(Y==label[i])][:,j]))
    return pdfs

#对一个样本进行预测
def Predect(X,p_y,pdfs,label,nf,nc):
    ups = np.zeros(nc)
    for j in range(nc): #对每个类，找到找到各个特征值对应的概率密度
        up = 1
        for n in range(nf):#循环求得概率密度
            x=pdfs[j][n][0]#类别j的第n个特征的pdf
            y=pdfs[j][n][1]
                
            index=np.nonzero(x>=X[n])[0]
            if len(index)==0:
                index=-1
            else:
                index = index[0]
            #print(i,n)
            up=up*y[index]#可能为0
        up = up*p_y[j]#将概率密度与p(y=i)相乘得到分子
        ups[j]=up#分母为所有分子的和，保存所有分子
    down = np.sum(ups)#分母
    p=ups*1.0/down#p(y=j|x=i),即当样本为i时类别为j的概率
    m = p.argmax(axis=0)#取概率最大值
    pre=label[m]
    return pre

#分类器，对一批样本预测
def Classifier(test,p_y,pdfs,label,nf,nc):
    #pre = np.zeros([len(test),nc])
    predict = np.zeros(len(test))
    for i in range(len(test)):
        predict[i]=Predect(test[i],p_y,pdfs,label,nf,nc)
    return predict




with open('predict_train_test_data.pickle', 'rb') as handle:
    data = pickle.load(handle,encoding='bytes')
    
    
trainX_10      = data[b'x_train_new']
trainY_10      = data[b'trainY']
testX_10       = data[b'x_test_new']
testY_10       = data[b'testY']





index=np.nonzero(trainY_10==0)
trainX1 = trainX_10[index]
trainY1 = trainY_10[index]

label = np.unique(trainY1)
nc = label.shape[0]
nf = trainX1.shape[1]
p_y=PDL(trainY1)
pdfs=PDF_x_Y(nc=nc,nf=nf,X=trainX1[:50],Y=trainY1[:50])

bw=0.08