#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 07:16:44 2021

@author: nmei

Title:Beyond category-supervision: instance-level contrastive learning models predict human visual system responses to objects
Link:https://www.biorxiv.org/content/10.1101/2020.06.15.153247v2.full.pdf

voxel selection based on the encoding results
"""

import os,gc,torch

import numpy  as np
import pandas as pd

from glob import glob

from nilearn.input_data import NiftiMasker
from nilearn.datasets   import load_mni152_template,load_mni152_brain_mask
from joblib             import Parallel,delayed
from sklearn            import preprocessing,metrics,model_selection as skms

from scipy.spatial             import distance
from nilearn.image             import new_img_like
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball
from torch import nn,optim

from utils import (load_event_files,
                   load_computational_features,
                   groupby_average,
                   searchlight_function_unit,
                   feature_normalize)

from utils_deep import (ridge,
                        ridge_train_valid)

def score_func(y, y_pred,tol = 1e-2,func = np.mean,is_train = True):
    # compute the raw R2 score for each voxel
    temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
    # find how many voxel can be explained by the computational model
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
    if is_train: # we want to find a balance between the variance explained and the positive voxels
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
    
    Inputs
    ------------
    input_image, 3D/4D Nifti object
        the averaged BOLD signal brain map
    computational_model_RDM, ndarray, scipy.spatial.distance.pdist
        the pair-wise correlation between each pair of unique words
    whole_brain_mask, 3D/4D Nifti object
        the standard whole brain mask
    sl_rad, int
        searchlight radius, in mm
    max_blk_edge, int
        unknown
    shape, Brainiak object
        the shape of the moving searchlight sphere used for extracting voxel values
    min_active_voxels_proportion, int or float
        unknown
    
    Output
    -------------
    global_outputs, ndarray
        X by Y by Z by 1
    """
    print('run RSA')
    sl = Searchlight(sl_rad                         = sl_rad, 
                      max_blk_edge                   = max_blk_edge, 
                      shape                          = shape,
                      min_active_voxels_proportion   = min_active_voxels_proportion,
                      )
    # this is where the searchlight will be moving on
    sl.distribute([np.asanyarray(input_image.dataobj)], 
                    np.asanyarray(whole_brain_mask.dataobj) == 1)
    # this is used by all the searchlights
    sl.broadcast(computational_model_RDM)
    # run searchlight algorithm
    global_outputs = sl.run_searchlight(searchlight_function_unit,
                                        pool_size = -1, # use all the CPUs
                                        )
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
    radius              = 10 # sphere radius
    device              = 'cpu' # the model will break my GPU
    dropout_rate        = 0.9
    batch_size          = 8
    learning_rate       = 1e-3
    patience            = 5
    epochs              = int(3e3)
    tol                 = 1e-4
    l2_term             = 1e-2
    print_train         = True # verbose
    working_dir         = f'../results/Searchlight_standard/{sub}'
    event_dir           = f'../data/Searchlight/{sub}'
    mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
    folder_name         = f'encoding_based_FS_RSA_{radius}mm'
    output_dir          = f'../results/{folder_name}'
    model_dir           = f'../models/{folder_name}'
    whole_brain_data    = np.sort(glob(os.path.join(working_dir,'*.nii.gz')))
    events              = np.sort(glob(os.path.join(event_dir,'while_brain_bold_stacked.csv')))
    whole_brain_mask    = load_mni152_brain_mask()
    example_func        = load_mni152_template()
    # make folder for the RSA maps
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # make folder for the encoding maps - as a basline for RSA
    if not os.path.exists(output_dir.replace(f'encoding_based_RSA_{radius}mm','encoding')):
        os.mkdir(output_dir.replace(f'encoding_based_RSA_{radius}mm','encoding'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(
            os.path.join(os.path.join(output_dir,
                                      f'{idx_sub}_{condition}_{model_name}.nii.gz'))):
        # prepare for loading the data
        masker              = NiftiMasker(whole_brain_mask,).fit() # maker
        df_events           = pd.concat([load_event_files(f,idx = 3) for f in events]) # change dir idx
        # load the data into memory - parallize the jobs
        res = Parallel(n_jobs = -1,verbose = 1,)(delayed(masker.transform)(**{
                'imgs':item}) for item in whole_brain_data)
        BOLD_array          = np.concatenate(res) # for encoding, n_samples x n_voxels
        del res # save some memory because we will need a lot
        # convert the byt data to string
        df_events['words']  = df_events['words'].apply(lambda x:x.lower())
        # load the 36 unique words
        unique_words        = np.load('../data/unique_words/words.npy').astype(str)
        # load the computational features - the 1st argument is for modification
        df_features         = load_computational_features('../results',model_name)
        
        # pick condition
        idx_condition       = df_events['context'] == condition
        brain_features      = BOLD_array[idx_condition]
        df_data             = df_events[idx_condition]
        del BOLD_array
        
        # leave one subject out cross validation
        idx_test            = df_data['sub'] == idx_sub # the left-out subject
        idx_train           = df_data['sub'] != idx_sub # the training subjects
        
        array_train         = brain_features[idx_train]
        df_data_train       = df_data[idx_train]
        features_train      = np.array([df_features[item] for item in df_data_train['words'].values])
        groups_train        = df_data_train['sub'].values
        
        array_test          = brain_features[idx_test]
        df_data_test        = df_data[idx_test]
        features_test       = np.array([df_features[item] for item in df_data_test['words'].values])
        del brain_features
        
        # leave one subject out cross validation partitioning
        cv                  = skms.LeaveOneGroupOut()
        for train,valid in cv.split(features_train,groups = groups_train):
            train,valid
        
        # scaler the y that is used in encoding model
        scaler_feature      = preprocessing.MinMaxScaler((0,1.)).fit(features_train)
        scaler_brain        = preprocessing.MinMaxScaler((0,1.)).fit(array_train)
        features_train_normalized = scaler_feature.transform(features_train)
        array_train_normalized = scaler_brain.transform(array_train)
        
        torch.manual_seed(12345)
        in_features,out_features = features_train.shape[1],array_train.shape[1]
        ridge_model = ridge(device = device,
                            dropout_rate = dropout_rate,
                            in_features = in_features,
                            out_features = out_features,
                            )
        #Loss function
        loss_func   = nn.BCEWithLogitsLoss()
        #Optimizer
        optimizer   = optim.Adam([params for params in ridge_model.parameters()],
                                  lr = learning_rate,
                                  weight_decay = l2_term)
        # train and validate the model
        if os.path.exists(os.path.join(model_dir,
                                       f'{condition}_{model_name}_{idx_sub}.pth')):
            ridge_model = torch.load(os.path.join(model_dir,
                                                  f'{condition}_{model_name}_{idx_sub}.pth'))
        else:
            ridge_model = ridge_train_valid(
                ridge_model,
                loss_func                   = loss_func,
                optimizer                   = optimizer,
                device                      = device,
                f_name                      = os.path.join(model_dir,
                                                           f'{condition}_{model_name}_{idx_sub}.pth'),
                features_train_normalized   = features_train_normalized,
                array_train_normalized      = array_train_normalized,
                train                       = train,
                valid                       = valid,
                epochs                      = epochs,
                tol                         = tol,
                patience                    = patience,
                batch_size                  = batch_size,
                l2_term                     = 0,# L2 is defined in the optimizer
                print_train                 = print_train,)
        
        #######################################################################
        with torch.no_grad():
            out_func = nn.Sigmoid()
            array_train_pred = out_func(ridge_model(
                    torch.from_numpy(features_train).float())).to('cpu').detach().numpy()
            encoding_score = metrics.r2_score(array_train_normalized,
                                              array_train_pred,
                                              multioutput = 'raw_values')
        # voxel secltion based on the encoding scores of the training data
        n_positive = np.sum(encoding_score > 0)
        if n_positive > 100:
            idx_voxels = np.where(encoding_score > 0)
        else:
            idx_voxels = np.argpartition(encoding_score,-300)[-300:]
        # encoding model results
        with torch.no_grad():
            out_func = nn.Sigmoid()
            array_test_pred     = out_func(ridge_model(
                torch.from_numpy(features_test).float())).to('cpu').detach().numpy() # we will use this for the RSA as well
        array_test_normalized   = scaler_brain.transform(array_test)
        encoding_score          = metrics.r2_score(array_test_normalized,
                                                   array_test_pred,
                                                   multioutput = 'raw_values')
        
        encoding_brain = masker.inverse_transform(encoding_score)
        encoding_brain.to_filename(os.path.join(
                output_dir.replace(f'encoding_based_RSA_{radius}mm','encoding'),
                f'{idx_sub}_{condition}_{model_name}.nii.gz'))
        
        # select the voxels
        array_test_normalized = array_test_normalized[:,idx_voxels]
        #######################################################################
        # now make sure we are working on the testing subject #################
        #######################################################################
        # we will correlate the predicted BOLD with the actual BOLD ###########
        #######################################################################
        
        # average predicted responses and the brain responeses for each word
        temp,df_average = groupby_average([array_test_pred,array_test_normalized],
                                          df_data_test.reset_index(drop = True),
                                          groupby = ['words'])
        
        BOLD_average        = temp[0] # 36 x n_voxels
        BOLD_pred_average   = temp[1] # 36 x n_voxels after selection
        
        # normalize the model features
        BOLD_pred_average   = feature_normalize(BOLD_pred_average)
        # RDMs of the model features
        RDM                 = distance.pdist(BOLD_pred_average,'correlation')
        BOLD_average        = masker.inverse_transform(BOLD_average)
        
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
    else:
        print(glob(os.path.join(output_dir.replace(f'encoding_based_RSA_{radius}mm','encoding'),
                           f'{idx_sub}_{condition}_{model_name}.nii.gz')))
        print(glob(os.path.join(output_dir,f'{idx_sub}_{condition}_{model_name}.nii.gz')))