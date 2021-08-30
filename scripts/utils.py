#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 08:49:35 2021

@author: nmei
"""
import os,gc

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

def convert_individual_space_to_standard_space(brain_map_individual_space,
                                               standard_brain,
                                               in_transformation_matrix_file,
                                               out_transformation_matrix_file,
                                               brain_map_standard_space,
                                               run_algorithm = False,
                                               ):
    """
    FSL FLIRT alogirthm called by nipype
    
    brain_map_individual_space: string or os.path.abspath
        the path of the brain map in individual space
    standard_brain: string or os.path.abspath
        the path of the MNI standard brain 2mm brain map
    in_transformation_matrix_file: string or os.path.abspath
        the path of the transformation matrix that convert individual space to standard space
    out_transformation_matrix_file: string or os.path.abspath
        an output transformation matrix
    brain_map_standard_space: string or os.path.abspath
        output file path
    run_algorithm: bool, default = False
    """
    from nipype.interfaces import fsl
    
    flt = fsl.FLIRT()
    flt.inputs.in_file          = os.path.abspath(brain_map_individual_space)
    flt.inputs.reference        = os.path.abspath(standard_brain)
    flt.inputs.in_matrix_file   = os.path.abspath(in_transformation_matrix_file)
    flt.inputs.out_matrix_file  = os.path.abspath(out_transformation_matrix_file)
    flt.inputs.out_file         = os.path.abspath(brain_map_standard_space)
    flt.inputs.output_type      = 'NIFTI_GZ'
    flt.inputs.apply_xfm        = True
    if run_algorithm:
        res = flt.run()
    else:
        res = flt.cmdline
    return res