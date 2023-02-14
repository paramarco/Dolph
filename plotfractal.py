# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:37:56 2023

@author: klio_ks
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
listPrice =[39.0,34.0,21.0,14.0,9.0,5.0]
listTime = [1.0,2.0,5.0,10.0,15.0,20.0]
vP=np.array(listPrice)
vT=np.array(listTime)
coeffs = np.polyfit(np.log(vT), np.log(vP), 3)
# print(coeffs[0])
plt.plot(vT,vP ,'g')
trendpoly = np.poly1d(coeffs) 
plt.plot(vT,trendpoly(vT),'r')
plt.show()