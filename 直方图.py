# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:20:43 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt #用于绘图的模块
font = {'family' : 'NSimSun,Times New Roman',   
        'weight' : 'normal',  
        'size'   : 10.05,  
        }  
plt.rc('font',**font) 
np.random.seed(1234)    #设置随机种子
N = 200  #随机产生的样本量
bins = 20
randnorm = np.random.normal(size = N)   #生成正态随机数

plt.figure(figsize=(5.2,3.9))
#直方图的个数用2个量级进行bins=sqrt(N)
#counts, bins, path = plt.hist(randnorm, bins = int(np.sqrt(N)), normed = True, color = 'blue')  #绘制直方图以上将直方图的频数和组距存放在counts和bins内。
#counts, bins, path = plt.hist(randnorm, bins = bins, normed = True, color = 'blue',label='直方图')
counts, bins, path = plt.hist(randnorm, bins = bins,  normed = True,rwidth=0.95,label='直方图')
#与正太分布密度函数进行比较
sigma = 1; mu = 0
norm_dist = (1/np.sqrt(2*sigma*np.pi))*np.exp(-((bins-mu)**2)/2)    #正态分布密度函数

plt.plot(bins,norm_dist,color = 'red',label='真实概率密度') #绘制正态分布密度函数图
plt.xlabel('特征取值')
plt.ylabel('概率密度')
plt.legend()
plt.show()