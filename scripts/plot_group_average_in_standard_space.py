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
from nilearn.image import concat_imgs,mean_img
from nilearn.surface import vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.input_data import NiftiMasker

from utils import nipype_fsl_randomise

from matplotlib import pyplot as plt

sns.set_context('paper')

radius              = 10
rerun_randomise     = False
folder_name         = f'RSA_basedline_average_{radius}mm_standard'
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
    concat_images   = concat_imgs(working_data)
    # mask the output before we conduct statistical inference
    data                = masker.transform(concat_images)
    image_for_randomise = masker.inverse_transform(data)
    image_for_plot      = mean_img(image_for_randomise)
    data                = masker.transform(image_for_plot)
    # exclude negative correlation coefficients
    data[0 > data]      = 0
    image_for_plot      = masker.inverse_transform(data)
    
    # left
    df['surf_mesh'              ].append(fsaverage.infl_left)
    df['bg_map'                 ].append(fsaverage.sulc_left)
    df['hemisphere'             ].append('left')
    df['title'                  ].append(f'{condition_map[condition]}, {model_name_map[model_name]}')
    df['stat_brain_map_standard'].append(image_for_plot)
    df['randomise_brain_map'    ].append(image_for_randomise)
    
    # right
    df['surf_mesh'              ].append(fsaverage.infl_right)
    df['bg_map'                 ].append(fsaverage.sulc_right)
    df['hemisphere'             ].append('right')
    df['title'                  ].append(None)
    df['stat_brain_map_standard'].append(image_for_plot)
    df['randomise_brain_map'    ].append(image_for_randomise)
    iterator.set_description(f'{condition_map[condition]}, {model_name_map[model_name]}')
df = pd.DataFrame(df)

# randomise
if rerun_randomise:
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir) 
    for ii,row in list(df.iterrows())[::2]:
        row
        maps_for_randomise = row['randomise_brain_map']
        condition,model_name = row['title'].split(', ')
        maps_for_randomise.to_filename(os.path.join(temp_dir,
                  f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}.nii.gz'))
        input_file = os.path.join(temp_dir,
                  f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}.nii.gz')
        mask_file = standard_brain_mask
        base_name = os.path.join(randomise_dir,
                  f'{condition.replace(" ","_")}_{model_name.replace(" ","_")}')
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

vmax        = .1
bottom,top  = 0.1,0.9
left,right  = 0.1,0.8
data        = np.random.uniform(0,vmax,size = (50,50))
# for making the colorbar
im          = plt.imshow(data,cmap = plt.cm.Reds,vmin = 0,vmax = vmax)

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
    bg_map          = row['bg_map']
    hemi            = row['hemisphere']
    title           = row['title']
    
    brain_map_in_surf = vol_to_surf(image_for_plot,surf_mesh,radius = radius,)
    plot_surf_stat_map(surf_mesh,
                       brain_map_in_surf,
                       bg_map           = bg_map,
                       threshold        = 0.005,
                       hemi             = hemi,
                       axes             = ax,
                       figure           = fig,
                       title            = title,
                       cmap             = plt.cm.bwr,
                       colorbar         = False,
                       vmax             = vmax,
                       symmetric_cbar   = 'auto',)
cbar_ax = fig.add_axes([0.92,bottom,0.01,top - bottom])
cbar    = fig.colorbar(im,cax = cbar_ax)
cbar.set_ticks(np.array([0,vmax]))
cbar.set_ticklabels(np.array([0,vmax],dtype = str))
fig.savefig(os.path.join(figure_group_average_dir,'group average.jpg'),
            bbox_inches = 'tight')
plt.close('all')









