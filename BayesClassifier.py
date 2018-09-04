# -*- coding: utf-8 -*-

import numpy as np
import pickle
from scipy import stats
#Probability distribution list
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
    #bw = kde.scotts_factor() * np.std(x)#带宽计算，使用
    bw=0.08
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