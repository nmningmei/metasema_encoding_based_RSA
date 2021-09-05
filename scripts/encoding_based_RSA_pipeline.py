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

from glob import glob

from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_template,load_mni152_brain_mask
from joblib import Parallel,delayed
from sklearn import linear_model,preprocessing,metrics,model_selection as skms
from sklearn.pipeline import make_pipeline

from scipy.spatial             import distance
from nibabel                   import load as load_fmri
from nilearn.image             import new_img_like
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball

from utils import (load_event_files,
                   load_computational_features,
                   groupby_average,
                   searchlight_function_unit,
                   feature_normalize)

def score_func(y, y_pred,tol = 1e-2,func = np.mean,is_train = True):
    temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
    n_positive  = np.sum(temp > tol)
    if n_positive > 0:
        score   = func(temp[temp > tol])
    else:
        score   = 0
    counter     = 0
    the_number  = n_positive
    while the_number > 0:
        the_number = the_number // 10
        counter += 1
    if is_train:
        return 0.5 * score + 0.5 * (n_positive / 10 ** counter)
    else:
        return score

scorer = metrics.make_scorer(score_func, greater_is_better = True,)

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
                   np.asanyarray(whole_brain_mask.dataobj) == 1)
    sl.broadcast(computational_model_RDM)
    # run searchlight algorithm
    global_outputs = sl.run_searchlight(searchlight_function_unit,
                                        pool_size = -1)
    return global_outputs

if __name__ == "__main__":
    for item in glob('core.*'):
        os.system(f'rm {item}')
    # parameters in the header
    sub                 = '*'
    idx_sub             = '123' # change sub
    condition           = 'read' # change condition
    model_name          = 'mobilenet' # change model_name
    n_jobs              = 6 # change n_jobs
    alpha_max           = 5 # change alpha
    radius              = 10
    working_dir         = f'../results/Searchlight_standard/{sub}'
    event_dir           = f'../data/Searchlight/{sub}'
    mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
    folder_name         = f'encoding_based_RSA_{radius}mm'
    output_dir          = f'../results/{folder_name}'
    whole_brain_data    = np.sort(glob(os.path.join(working_dir,'*.nii.gz')))
    events              = np.sort(glob(os.path.join(event_dir,'while_brain_bold_stacked.csv')))
    whole_brain_mask    = load_mni152_brain_mask()
    example_func        = load_mni152_template()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # prepare for loading the data
    masker              = NiftiMasker(whole_brain_mask,).fit()
    df_events           = pd.concat([load_event_files(f) for f in events])
    res = Parallel(n_jobs = -1,verbose = 1,)(delayed(masker.transform)(**{
            'imgs':item}) for item in whole_brain_data)
    BOLD_array          = np.concatenate(res) # for encoding
    del res
    df_events['words']  = df_events['words'].apply(lambda x:x.lower())
    unique_words        = np.load('../data/unique_words/words.npy').astype(str)
    df_features         = load_computational_features(model_name)
    
    # pick condition
    idx_condition       = df_events['context'] == condition
    brain_features      = BOLD_array[idx_condition]
    df_data             = df_events[idx_condition]
    del BOLD_array
    
    # leave one subject out cross validation
    idx_test            = df_data['sub'] == idx_sub
    idx_train           = df_data['sub'] != idx_sub
    
    array_train         = brain_features[idx_train]
    df_data_train       = df_data[idx_train]
    features_train      = np.array([df_features[item] for item in df_data_train['words'].values])
    groups_train        = df_data_train['sub'].values
    array_test          = brain_features[idx_test]
    df_data_test        = df_data[idx_test]
    features_test       = np.array([df_features[item] for item in df_data_test['words'].values])
    del brain_features
    
    cv                  = skms.LeaveOneGroupOut()
    linear_reg          = linear_model.Ridge(fit_intercept  = True,
                                             normalize      = True,
                                             random_state   = 12345,
                                             )
    reg                 = make_pipeline(preprocessing.MinMaxScaler((-1,1)),
                                        linear_reg,
                                        )
    scaler_brain        = preprocessing.MinMaxScaler((-1,1)).fit(array_train)
    array_train_normalized = scaler_brain.transform(array_train)
    param_grid          = {'ridge__alpha':np.logspace(2,alpha_max,alpha_max - 2 + 1)}
    gc.collect()
    grid_search         = skms.GridSearchCV(reg,
                                            param_grid,
                                            scoring = scorer,
                                            cv      = skms.LeaveOneGroupOut(),
                                            verbose = 1,
                                            n_jobs  = n_jobs,
                                    )
    grid_search.fit(features_train,array_train_normalized,groups_train)
    gc.collect()
    
    array_test_pred = grid_search.predict(features_test)
    
    temp,df_average = groupby_average([array_test_pred,features_test],
                                      df_data_test.reset_index(drop = True),
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
    
    res  = new_img_like(example_func,np.array(res,  dtype = np.float),)
    
    # save
    res.to_filename(os.path.join(output_dir, f'{idx_sub}_{condition}_{model_name}.nii.gz'))























