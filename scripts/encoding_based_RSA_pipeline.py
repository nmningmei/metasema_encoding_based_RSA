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

from glob import glob

from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_template,load_mni152_brain_mask
from nilearn.image import concat_imgs

from sklearn import linear_model,preprocessing,metrics,model_selection as skms
from sklearn.pipeline import make_pipeline

from utils import load_event_files,load_computational_features

for item in glob('core.*'):
    os.system(f'rm {item}')
# parameters in the header
sub                 = '*'
model_name          = 'mobilenet'
working_dir         = f'../results/Searchlight_standard/{sub}'
event_dir           = f'../data/Searchlight/{sub}'
mask_dir            = f'../data/masks_and_transformation_matrices/{sub}'
whole_brain_data    = np.sort(glob(os.path.join(working_dir,'*.nii.gz')))
events              = np.sort(glob(os.path.join(event_dir,'while_brain_bold_stacked.csv')))
whole_brain_mask    = load_mni152_brain_mask()
example_func        = load_mni152_template()

# prepare for loading the data
masker              = NiftiMasker(whole_brain_mask,).fit()
concat_BOLD         = concat_imgs(whole_brain_data) # for RSA
df_events           = pd.concat([load_event_files(f) for f in events])
BOLD_array          = masker.transform(concat_BOLD) # for encoding
df_events['words']  = df_events['words'].apply(lambda x:x.lower())
unique_words        = np.load('../data/unique_words/words.npy').astype(str)
df_features         = load_computational_features(model_name)
condition           = 'read'

# leave one subject out cross validation
idx_sub             = '123' # change here
idx_test            = df_events['sub'] == idx_sub
idx_train           = df_events['sub'] != idx_sub



























