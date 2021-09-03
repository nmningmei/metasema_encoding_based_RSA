#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:04:31 2021

@author: nmei
"""

import os
import numpy as np
from glob import glob
from nilearn.input_data import NiftiMasker
from utils import convert_individual_space_to_standard_space
from joblib import Parallel,delayed

for item in glob('core.*'):
    os.system(f'rm {item}')

folder_name             = 'Searchlight'
working_dir             = f'../data/{folder_name}'
working_data            = glob(os.path.join(working_dir,'*','*.npy'))
standard_brain          = '../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
transformation_dir      = '../data/masks_and_transformation_matrices'
out_transformation_dir  = '../results/transformation'
output_dir              = f'../results/{folder_name}_standard'
for f in [out_transformation_dir,output_dir]:
    if not os.path.exists(f):
        os.mkdir(f)

def _proc(brain_map_file):
    _,_,_,sub,filename = brain_map_file.split('/')
    in_transformation_matrix_file = os.path.join(
            transformation_dir,
            sub,
            'reg',
            'example_func2standard.mat')
    mask = os.path.join(
            transformation_dir,
            sub,
            'mask.nii.gz')
    masker = NiftiMasker(mask,).fit()
    brain_map = masker.inverse_transform(np.load(brain_map_file))
    input_brain_map_file = os.path.join(
            'temp',
            f'{sub}.nii.gz')
    brain_map.to_filename(input_brain_map_file)
    out_transformation_matrix_file = os.path.join(
            out_transformation_dir,
            f'{sub}.mat')
    brain_map_standard_space = brain_map_file.replace('data','results',).replace(folder_name,f'{folder_name}_standard').replace('.npy','.nii.gz')
    if not os.path.exists(os.path.join(*brain_map_standard_space.split('/')[:-1])):
        os.mkdir(os.path.join(*brain_map_standard_space.split('/')[:-1]))
    res = convert_individual_space_to_standard_space(
            brain_map_individual_space      = input_brain_map_file,
            standard_brain                  = standard_brain,
            in_transformation_matrix_file   = in_transformation_matrix_file,
            out_transformation_matrix_file  = out_transformation_matrix_file,
            brain_map_standard_space        = brain_map_standard_space,
            run_algorithm                   = True,
            )
    return res

res = Parallel(n_jobs = -1,verbose = 1)(delayed(_proc)(**{
        'brain_map_file':brain_map_file}) for brain_map_file in working_data)

for item in glob('core.*'):
    os.system(f'rm {item}')