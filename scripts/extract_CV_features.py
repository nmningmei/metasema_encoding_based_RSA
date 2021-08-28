#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:02:36 2021

@author: nmei
"""

import os,torch,utils_deep

from glob import glob

if __name__ == "__main__":
    
    image_dir       = '../data/images_for_feature_extraction'
    image_paths     = glob(os.path.join(image_dir,'*','*','*.jpg'))
    fine_tune_dir   = '../data/101_ObjectCategories_grayscaled'
    fine_tune_size  = 96 if '101' in fine_tune_dir else 2
    output_dir      = '../data/cv_features'
    figure_dir      = '../figures'
    
    model_names     = ['vgg19_bn','mobilnetv2','resnet50']
    image_resize    = 128
    batch_size      = 32
    hidden_size     = 300 # to compare with word2vec models
    learning_rate   = 1e-3
    n_epochs        = int(3e3)
    patience        = 5
    noise_level     = None
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model_name in model_names:
        print(model_name)
        
        model_to_train = utils_deep.build_model(
            pretrain_model_name = model_name,
            hidden_units = hidden_size,
            hidden_activation_name = 'selu',
            hidden_dropout = 0.5,
            output_units = fine_tune_size,
            )
        loss_func,optimizer = utils_deep.createLossAndOptimizer(model_to_train,
                                                                learning_rate = learning_rate)
        model_to_train = utils_deep.train_and_validation(
        model_to_train,
        f_name              = f'../models/{model_name}.pt',
        optimizer           = optimizer,
        image_resize        = image_resize,
        device              = device,
        batch_size          = batch_size,
        n_epochs            = n_epochs,
        print_train         = True,
        patience            = patience,
        train_root          = fine_tune_dir,
        valid_root          = fine_tune_dir,
        noise_level         = noise_level,
        )
        
        adsf
        