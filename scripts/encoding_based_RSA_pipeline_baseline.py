#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 07:16:44 2021

@author: nmei

Title:Beyond category-supervision: instance-level contrastive learning models predict human visual system responses to objects
Link:https://www.biorxiv.org/content/10.1101/2020.06.15.153247v2.full.pdf


"""

import os
import utils

import numpy  as np
import pandas as pd

from nilearn.input_data import NiftiMasker

# parameters in the header
sub                 = '123'
cv_model_name       = 'vgg19'
w2v_model_name      = 'fast text'
working_dir         = f'../data/Searchlight/{sub}'
mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
whole_brain_data    = os.path.join(working_dir,'while_brain_bold_stacked.npy')
events              = os.path.join(working_dir,'while_brain_bold_stacked.csv')
combined_mask       = os.path.join(mask_dir,'mask.nii.gz')
example_func        = os.path.join(mask_dir,'example_func.nii.gz')
df_cv_features      = pd.read_csv(os.path.join('../results/computer_vision_features_no_background_caltech',
                                               f'{cv_model_name}.csv')
                                  )
df_w2v_features     = pd.read_csv(os.path.join('../results/word2vec_features',
                                               f'{w2v_model_name}.csv')
                                  )

# prepare for loading the data
masker              = NiftiMasker(combined_mask,).fit()
BOLD_array          = np.load(whole_brain_data)
df_events           = pd.read_csv(events)
df_events['words']  = df_events['words'].apply(lambda x:x.lower())
unique_words        = np.load('../data/unique_words/words.npy')
condition           = 'read'

# select the data based on the condition
idx_condition   = df_events['context'] == condition
BOLD_condition  = BOLD_array[idx_condition]
df_condition    = df_events[idx_condition]
cv_features     = np.array([df_cv_features[word] for word in df_condition['words']])
w2v_features    = np.array([df_w2v_features[word] for word in df_condition['words']])



























