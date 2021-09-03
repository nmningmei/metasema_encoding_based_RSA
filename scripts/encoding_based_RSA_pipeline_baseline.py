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
                   searchlight_function_unit,
                   load_computational_features
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
    for sub                 in os.listdir('../data/Searchlight'):
        for condition           in ['read','reenact']:
            model_name          = 'mobilenet' # change here
            radius              = 10 # define your RSA radius
            working_dir         = f'../data/Searchlight/{sub}'
            mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
            whole_brain_data    = os.path.join(working_dir,'while_brain_bold_stacked.npy')
            events              = os.path.join(working_dir,'while_brain_bold_stacked.csv')
            whole_brain_mask    = os.path.join(mask_dir,'mask.nii.gz')
            example_func        = os.path.join(mask_dir,'example_func.nii.gz')
            df_features         = load_computational_features(model_name)
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
            features        = np.array([df_features[word] for word in df_condition['words']])
            
            # average the data for RSA
            temp,df_condition_average = groupby_average([BOLD_condition,
                                                         features,],
                                                         df_condition,
                                                         groupby = ['words'])
            BOLD_average    = temp[0]
            features        = temp[1]
            
            # normalize the model features
            features        = feature_normalize(features)
            # RDMs of the model features
            RDM             = distance.pdist(features,'correlation')
            BOLD_average    = masker.inverse_transform(BOLD_average)
            
            # perform RSA on the brain
            gc.collect()
            res             = _searchligh_RSA(BOLD_average,
                                              RDM,
                                              whole_brain_mask,
                                              sl_rad = radius,
                                              )
            gc.collect()
            
            res  = new_img_like(load_fmri(example_func),np.array(res,  dtype = np.float),)
            
            # save
            res.to_filename(os.path.join(output_dir, f'{sub}_{condition}_{model_name}.nii.gz'))
