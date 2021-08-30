#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 06:39:27 2021

@author: nmei
"""

import os

from glob import glob
from utils import convert_individual_space_to_standard_space

radius                  = 10
folder_name             = f'RSA_basedline_average_{radius}mm'
working_dir             = f'../results/{folder_name}'
working_data            = glob(os.path.join(working_dir,'*.nii.gz'))
standard_brain          = '../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
transformation_dir      = '../data/masks_and_transformation_matrices'
out_transformation_dir  = '../results/transformation'
output_dir              = f'../results/{folder_name}_standard'
for f in [out_transformation_dir,output_dir]:
    if not os.path.exists(f):
        os.mkdir(f)

for brain_map_file in working_data:
    sub,condition,model_name = brain_map_file.split('/')[-1].split('.')[0].split('_')
    in_transformation_matrix_file = os.path.join(
            transformation_dir,
            sub,
            'reg',
            'example_func2standard.mat')
    out_transformation_matrix_file = os.path.join(
            out_transformation_dir,
            f'{sub}_{condition}_{model_name}.mat')
    brain_map_standard_space = os.path.join(
            output_dir,
            f'{sub}_{condition}_{model_name}_standard.nii.gz')
    res = convert_individual_space_to_standard_space(
            brain_map_individual_space = brain_map_file,
            standard_brain = standard_brain,
            in_transformation_matrix_file = in_transformation_matrix_file,
            out_transformation_matrix_file = out_transformation_matrix_file,
            brain_map_standard_space = brain_map_standard_space,
            run_algorithm = True,
            )
    






































