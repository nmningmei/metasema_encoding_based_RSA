#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 08:49:35 2021

@author: nmei
"""

import numpy as np
import pandas as pd

def groupby_average(fmri,df,groupby = ['trials'],axis = 0):
    BOLD_average = np.array([np.mean(fmri[df_sub.index],axis) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([df_sub for ii,df_sub in df.groupby(groupby)])
    return BOLD_average,df_average