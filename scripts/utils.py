#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 08:49:35 2021

@author: nmei
"""

import numpy as np
import pandas as pd

def groupby_average(list_of_arrays,df,groupby = ['trials'],axis = 0):
    place_holder = {ii:[] for ii in range(len(list_of_arrays))}
    for ii,array in enumerate(list_of_arrays):
        place_holder[ii] = np.array([np.mean(array[df_sub.index],axis) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([df_sub.iloc[0,:].to_frame().T for ii,df_sub in df.groupby(groupby)])
    return place_holder,df_average