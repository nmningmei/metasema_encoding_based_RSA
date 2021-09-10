#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 08:27:21 2021

@author: nmei

Thi script does two things:
    1. plot the group-averaged data on inflated brain
    2. perform randomise on the group-level

"""

import os

import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from tqdm import tqdm
from itertools import product
from nilearn.surface import vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.input_data import NiftiMasker

from matplotlib import pyplot as plt
from utils import load_data_for_randomise

sns.set_context('paper')
for item in glob('core.*'):
    os.system(f'rm {item}')

radius              = 10
folder_name         = f'RSA_basedline_average_{radius}mm_standard' # we change this accordingly
# RSA_basedline_average_{radius}mm_standard
# encoding_based_RSA_{radius}mm
# encoding
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

randomise_dir               = f'../results/{folder_name}_randomise' # this is a temporal folder
figure_group_average_dir    = f'../figures/{folder_name}_group_average'
figure_stat_dir             = f'../figures/{folder_name}_randomise'
for f in [randomise_dir,figure_group_average_dir,figure_stat_dir]:
    if not os.path.exists(f):
        os.mkdir(f)

df = dict(surf_mesh                 = [],
          stat_mesh                 = [],
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
                                   return_tanh = False,)
    data = np.squeeze(data)
    image_for_randomise = masker.inverse_transform(data)
#    data_average = np.mean(data, 0)
#    data_average[0 > data_average] = 0
    image_for_plot      = masker.inverse_transform(np.mean(data,axis = 0))
    
    # left
    df['surf_mesh'              ].append(fsaverage.infl_left)
    df['stat_mesh'              ].append(fsaverage.pial_left)
    df['bg_map'                 ].append(fsaverage.sulc_left)
    df['hemisphere'             ].append('left')
    df['title'                  ].append(f'{condition_map[condition]}, {model_name_map[model_name]}')
    df['stat_brain_map_standard'].append(image_for_plot)
    df['randomise_brain_map'    ].append(image_for_randomise)
    
    # right
    df['surf_mesh'              ].append(fsaverage.infl_right)
    df['stat_mesh'              ].append(fsaverage.pial_right)
    df['bg_map'                 ].append(fsaverage.sulc_right)
    df['hemisphere'             ].append('right')
    df['title'                  ].append(None)
    df['stat_brain_map_standard'].append(image_for_plot)
    df['randomise_brain_map'    ].append(image_for_randomise)
    iterator.set_description(f'{condition_map[condition]}, {model_name_map[model_name]}')
df = pd.DataFrame(df)

# randomise
maps_randomise = glob(os.path.join(randomise_dir,'*tfce_corrp_tstat1.nii.gz'))
temp = []
for ii,row in list(df.iterrows())[::2]:
    condition,model_name = row['title'].split(', ')
    map_randomise = [item for item in maps_randomise if\
                     (condition.replace(" ","_") in item)\
                     and (model_name.replace(" ","_") in item)][0]
    pvalue      = masker.fit_transform(map_randomise)[0]
    # in randomise, p values are saved as 1 - P
#    pvalue_     = -np.log(pvalue)
#    pvalue_[pvalue_ == np.inf] = 0
    p_to_plot   = masker.inverse_transform(pvalue)
    temp.append(p_to_plot)
df['randomise_maps'] = np.repeat(temp,2)

vmax        = None
bottom,top  = 0.1,0.9
left,right  = 0.1,0.8

plt.close('all')
print('plotting in standard space')
fig,axes = plt.subplots(figsize     = (4 * 4,6 * 3),
                        nrows       = 6, # determined by models tested
                        ncols       = 4, # because I want to plot both left and right brain
                        subplot_kw  = {'projection':'3d'},
                        )
for ax,(ii_row,row) in zip(axes.flatten(),df.iterrows()):
    image_for_plot  = row['stat_brain_map_standard']
    surf_mesh       = row['surf_mesh']
    stat_mesh       = row['stat_mesh']
    bg_map          = row['bg_map']
    hemi            = row['hemisphere']
    title           = row['title']
    
    brain_map_in_surf = vol_to_surf(image_for_plot,stat_mesh,radius = 2,)
    plot_surf_stat_map(surf_mesh,
                       brain_map_in_surf,
                       bg_map           = bg_map,
                       threshold        = 1e-6,
                       hemi             = hemi,
                       axes             = ax,
                       figure           = fig,
                       title            = title,
                       cmap             = plt.cm.bwr,
                       colorbar         = True,
                       vmax             = vmax,
                       symmetric_cbar   = 'auto',)
fig.savefig(os.path.join(figure_group_average_dir,'group average.jpg'),
            bbox_inches = 'tight')
plt.close('all')

vmax = 1
plt.close('all')
print('plotting in standard space')
fig,axes = plt.subplots(figsize     = (4 * 4,6 * 3),
                        nrows       = 6, # determined by models tested
                        ncols       = 4, # because I want to plot both left and right brain
                        subplot_kw  = {'projection':'3d'},
                        )
for ax,(ii_row,row) in zip(axes.flatten(),df.iterrows()):
    image_for_plot  = row['randomise_maps']
    surf_mesh       = row['surf_mesh']
    stat_mesh       = row['stat_mesh']
    bg_map          = row['bg_map']
    hemi            = row['hemisphere']
    title           = row['title']
    
    brain_map_in_surf = vol_to_surf(image_for_plot,stat_mesh,radius = 2,)
    plot_surf_stat_map(surf_mesh,
                       brain_map_in_surf,
                       bg_map           = bg_map,
                       threshold        = 1 - 0.05,
                       hemi             = hemi,
                       axes             = ax,
                       figure           = fig,
                       title            = title,
                       cmap             = plt.cm.bwr,
                       colorbar         = True,
                       vmax             = vmax,
                       symmetric_cbar   = 'auto',)
fig.savefig(os.path.join(figure_stat_dir,'group average p values.jpg'),
            bbox_inches = 'tight')
plt.close('all')







