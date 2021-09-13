#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 02:50:18 2021

@author: nmei
"""

import os

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from itertools import product
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.input_data import NiftiMasker
from utils import nipype_fsl_randomise,load_data_for_randomise


radius              = 10
rerun_randomise     = False
folder_name         = f'decoding_based_RSA_{radius}mm' # we change this accordingly
# RSA_basedline_average_{radius}mm_standard
# encoding_based_RSA_{radius}mm
# encoding
# decoding_based_RSA_{radius}mm
working_dir         = f'../results/{folder_name}'
standard_brain_mask = '../data/standard_brain/MNI152_T1_2mm_brain_mask_dil.nii.gz'
conditions          = ['read','reenact']
model_names         = ['vgg19','mobilenet','resnet50','fasttext','glove','word2vec']
condition_map       = {'read':      'Shallow process',
                       'reenact':   'Deep process'}
model_name_map      = {'vgg19':     'VGG19',
                       'mobilenet': 'MobileNetV2',
                       'resnet50':  'ResNet50',
                       'fasttext':  'Fast Text',
                       'glove':     'GloVe',
                       'word2vec':  'Word2Vec',}

iterator    = tqdm(np.array(list(product(*[model_names,conditions]))))
masker      = NiftiMasker(standard_brain_mask,).fit()
fsaverage   = fetch_surf_fsaverage()

randomise_dir               = f'../results/{folder_name}_randomise'
for f in [randomise_dir,]:
    if not os.path.exists(f):
        os.mkdir(f)

df = dict(surf_mesh                 = [],
          bg_map                    = [],
          hemisphere                = [],
          title                     = [],
          stat_brain_map_standard   = [],
          randomise_brain_map       = [],
          )
for (model_name,condition) in iterator:
    working_data    = np.sort(glob(os.path.join(working_dir,
                                                f'*{condition}*{model_name}*nii.gz')
                                    )
                                )
    data = load_data_for_randomise(working_data,
                                   folder_name,
                                   masker,
                                   return_tanh = True,)
    data = np.squeeze(data)
    image_for_randomise = masker.inverse_transform(data)
    image_for_plot      = masker.inverse_transform(np.mean(data,axis = 0))
    # left
    df['surf_mesh'              ].append(fsaverage.infl_left)
    df['bg_map'                 ].append(fsaverage.sulc_left)
    df['hemisphere'             ].append('left')
    df['title'                  ].append(f'{condition_map[condition]}, {model_name_map[model_name]}')
    df['stat_brain_map_standard'].append(image_for_plot)
    df['randomise_brain_map'    ].append(image_for_randomise)
    iterator.set_description(f'{condition} {model_name} {data.shape}')
    
df = pd.DataFrame(df)

# randomise

temp_dir = 'temp'
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

idx = 4 # change idx
row = df.iloc[idx,:]
maps_for_randomise = row['randomise_brain_map']
condition,model_name = row['title'].split(', ')
maps_for_randomise.to_filename(os.path.join(temp_dir,
          f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}.nii.gz'))
input_file = os.path.join(temp_dir,
          f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}.nii.gz')
mask_file = standard_brain_mask
base_name = os.path.join(randomise_dir,
          f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}')
if not os.path.exists(base_name + '_tfce_corrp_tstat1.nii.gz'):
    nipype_fsl_randomise(input_file,
                         mask_file,
                         base_name,
                         tfce                   = True,
                         var_smooth             = 6,
                         demean                 = False,
                         one_sample_group_mean  = True,
                         n_permutation          = int(1e4),
                         quiet                  = True,
                         run_algorithm          = True,)