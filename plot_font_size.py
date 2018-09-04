# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:30:29 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
#类内单样本增量学习实验

import matplotlib.pyplot as plt
from math import *

font = {#'family' : 'Times New Roman',  
        'family' :'NSimSun,Times New Roman',
        'weight' : 'normal',  
        'size'   : 10.0,  
        }  
plt.rc('font',**font) 

l=[]
for i in range(0,8,1):
    l.append(i-7)
    
plt.figure(figsize=(5.2,3.9))
plt.plot(l)
plt.xlabel('个')
plt.show()