#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 08:49:35 2021

@author: nmei
"""

import numpy  as np
import pandas as pd

from scipy.spatial import distance
from scipy.stats   import spearmanr

def groupby_average(list_of_arrays,df,groupby = ['trials'],axis = 0):
    """
    To compute the average groupby the dataframe
    list_of_arrays: list of ndarrays
    df: pandas.DataFrame,
        dataframe object
    groupby: str or list of strings
    """
    place_holder = {ii:[] for ii in range(len(list_of_arrays))}
    for ii,array in enumerate(list_of_arrays):
        place_holder[ii] = np.array([np.mean(array[df_sub.index],axis) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([df_sub.iloc[0,:].to_frame().T for ii,df_sub in df.groupby(groupby)])
    return place_holder,df_average

def feature_normalize(features):
    """
    Normalize individual row of the matrix
    """
    features = features - features.mean(1).reshape(-1,1)
    return features


def searchlight_function_unit(shpere_bold_signals, mask, myrad, broadcast_variable):
    """
    shpere_bold_signals: BOLD
    mask: mask array
    myrad: not use
    broadcast_variable: label -- features RDM
    """
    BOLD    = shpere_bold_signals[0][mask,:].T.copy()
    RDM_y   = broadcast_variable.copy()
#    print(BOLD.shape)
    # pearson correlation
    RDM_X   = distance.pdist(feature_normalize(BOLD),'correlation')
    D,p     = spearmanr(RDM_X ,RDM_y)
#    print(D)
    return D