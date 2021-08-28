#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 07:16:44 2021

@author: nmei

Title:Beyond category-supervision: instance-level contrastive learning models predict human visual system responses to objects
Link:https://www.biorxiv.org/content/10.1101/2020.06.15.153247v2.full.pdf


"""

import os

import numpy  as np
import pandas as pd

from nilearn.input_data import NiftiMasker

from sklearn import linear_model,preprocessing,metrics,model_selection as skms
from sklearn.pipeline import make_pipeline

# parameters in the header
sub                 = '123'
working_dir         = f'../data/Searchlight/{sub}'
mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
whole_brain_data    = os.path.join(working_dir,'while_brain_bold_stacked.npy')
events              = os.path.join(working_dir,'while_brain_bold_stacked.csv')
combined_mask       = os.path.join(mask_dir,'mask.nii.gz')
example_func        = os.path.join(mask_dir,'example_func.nii.gz')

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

for word in unique_words:
    word = str(word,'UTF-8')
    idx_train = df_condition['words'] != word
    idx_test = df_condition['words'] == word



























