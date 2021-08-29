#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 07:16:44 2021

@author: nmei

Title:Beyond category-supervision: instance-level contrastive learning models predict human visual system responses to objects
Link:https://www.biorxiv.org/content/10.1101/2020.06.15.153247v2.full.pdf


"""

import os,gc

import numpy  as np
import pandas as pd

from scipy.spatial             import distance
from nibabel                   import load as load_fmri
from nilearn.image             import new_img_like
from nilearn.input_data        import NiftiMasker
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball
# addon
from utils import (groupby_average,
                   feature_normalize,
                   searchlight_function_unit
                   )

def _searchligh_RSA(input_image,
                    computational_model_RDM,
                    whole_brain_mask,
                    sl_rad                          = 9,
                    max_blk_edge                    = 9 - 1,
                    shape                           = Ball,
                    min_active_voxels_proportion    = 0,
                    ):
    """
    This is function is defined here is because the complex environment
    settings where Brainiak is installed
    """
    sl = Searchlight(sl_rad                         = sl_rad, 
                     max_blk_edge                   = max_blk_edge, 
                     shape                          = shape,
                     min_active_voxels_proportion   = min_active_voxels_proportion,
                     )
    sl.distribute([np.asanyarray(input_image.dataobj)], 
                   np.asanyarray(load_fmri(whole_brain_mask).dataobj) == 1)
    sl.broadcast(computational_model_RDM)
    # run searchlight algorithm
    global_outputs = sl.run_searchlight(searchlight_function_unit,
                                        pool_size = -1)
    return global_outputs

if __name__ == "__main__":
    # parameters in the header
    sub                 = '123'
    cv_model_name       = 'vgg19'
    w2v_model_name      = 'fasttext'
    condition           = 'read'
    radius              = 10
    working_dir         = f'../data/Searchlight/{sub}'
    mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
    whole_brain_data    = os.path.join(working_dir,'while_brain_bold_stacked.npy')
    events              = os.path.join(working_dir,'while_brain_bold_stacked.csv')
    whole_brain_mask    = os.path.join(mask_dir,'mask.nii.gz')
    example_func        = os.path.join(mask_dir,'example_func.nii.gz')
    df_cv_features      = pd.read_csv(os.path.join('../results/computer_vision_features_no_background_caltech',
                                                   f'{cv_model_name}.csv')
                                      )
    df_w2v_features     = pd.read_csv(os.path.join('../results/word2vec_features',
                                                   f'{w2v_model_name}.csv')
                                      )
    output_folder_name  = f'RSA_basedline_average_{radius}mm'
    output_dir          = f'../results/{output_folder_name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # prepare for loading the data
    masker              = NiftiMasker(whole_brain_mask,).fit()
    BOLD_array          = np.load(whole_brain_data)
    df_events           = pd.read_csv(events)
    df_events['words']  = df_events['words'].apply(lambda x:x.lower())
    unique_words        = np.load('../data/unique_words/words.npy')
    
    
    # select the data based on the condition
    idx_condition   = df_events['context'] == condition
    BOLD_condition  = BOLD_array[idx_condition]
    df_condition    = df_events[idx_condition].reset_index(drop = False)
    # we need to negate the values is we use log softmax during training
    cv_features     = np.array([-df_cv_features[word] for word in df_condition['words']])
    w2v_features    = np.array([df_w2v_features[word] for word in df_condition['words']])
    
    # average the data for RSA
    temp,df_condition_average = groupby_average([BOLD_condition,
                                                 cv_features,
                                                 w2v_features],
                                                 df_condition,
                                                 groupby = ['words'])
    BOLD_average    = temp[0]
    cv_features     = temp[1]
    w2v_features    = temp[2]
    
    # normalize the model features
    cv_features     = feature_normalize(cv_features)
    w2v_features    = feature_normalize(w2v_features)
    
    # RDMs of the model features
    RDM_cv          = distance.pdist(cv_features,'correlation')
    RDM_w2v         = distance.pdist(w2v_features,'correlation')
    
    BOLD_average = masker.inverse_transform(BOLD_average)
    
    # perform RSA on the brain
    gc.collect()
    map_cv  = _searchligh_RSA(BOLD_average,
                              RDM_cv,
                              whole_brain_mask,
                              sl_rad = radius,
                              )
    gc.collect()
    map_w2v = _searchligh_RSA(BOLD_average,
                              RDM_w2v,
                              whole_brain_mask,
                              sl_rad = radius,
                              )
    gc.collect()
    
    map_cv  = new_img_like(load_fmri(example_func),np.array(map_cv,  dtype = np.float),)
    map_w2v = new_img_like(load_fmri(example_func),np.array(map_w2v, dtype = np.float),)
    
    # save
    map_cv.to_filename(os.path.join(output_dir, f'{sub}_{condition}_{cv_model_name}.nii.gz'))
    map_w2v.to_filename(os.path.join(output_dir,f'{sub}_{condition}_{w2v_model_name}.nii.gz'))
    
#    map_cv_positive = masker.transform(map_cv)
#    map_cv_positive[0 > map_cv_positive] = 0
#    map_cv_positive = masker.inverse_transform(map_cv_positive)
#    
#    map_w2v_positive = masker.transform(map_w2v)
#    map_w2v_positive[0 > map_w2v_positive] = 0
#    map_w2v_positive = masker.inverse_transform(map_w2v_positive)
    

















