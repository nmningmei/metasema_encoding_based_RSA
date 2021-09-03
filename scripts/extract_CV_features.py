#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:02:36 2021

@author: nmei
"""

import os,torch

from glob       import glob
from matplotlib import pyplot as plt
from scipy.spatial import distance

from utils import groupby_average

from utils_deep import (
    createLossAndOptimizer,
    build_model,
    train_and_validation,
    extract_cv_features
    )
import numpy   as np
import pandas  as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

if __name__ == "__main__":
    
    image_dir       = '../data/images_for_feature_extraction'
    image_paths     = glob(os.path.join(image_dir,'*','*','*.jpg'))
    fine_tune_dir   = '../data/101_ObjectCategories'
    fine_tune_size  = 96 if '101' in fine_tune_dir else 2
    figure_dir      = '../figures/cv_features' # save the RDM image
    cv_dir          = '../results/computer_vision_features_no_background_caltech' # save the representations
    
    model_names     = ['vgg19','mobilenet','resnet50']
    image_resize    = 128
    batch_size      = 8
    hidden_size     = 300 # to compare with word2vec models
    learning_rate   = 5e-4 # higher than usual learning rate
    n_epochs        = int(3e3)
    patience        = 5
    noise_level     = None
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model_name in model_names:
        print(model_name)
        # train the model
        torch.manual_seed(12345)
        np.random.seed(12345)
        model_to_train = build_model(
            pretrain_model_name     = model_name,
            hidden_units            = hidden_size,
            hidden_activation_name  = 'selu',
            hidden_dropout          = 0.,
            output_units            = fine_tune_size,
            )
        loss_func,optimizer = createLossAndOptimizer(model_to_train,
                                                     learning_rate = learning_rate)
        model_to_train = train_and_validation(
        model_to_train,
        f_name              = f'../models/{model_name}.pt',
        loss_func           = loss_func,
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
        
        # freeze the trained model
        model_to_train.to('cpu')
        model_to_train.eval()
        for params in model_to_train.parameters():params.requires_grad == False
        
        # extract the features
        df,features = extract_cv_features(
            model_to_train,
            image_dir,
            image_resize        = image_resize,
            noise_level         = noise_level,
            do_augmentations    = False,
            )
        features_average,df_average = groupby_average([features], df,groupby=['labels'])
        df_average          = df_average.reset_index(drop = True)
        idx_sort            = list(df_average.sort_values(['targets','labels']).index)
        features_average    = features_average[0][idx_sort]
        df_average          = df_average.sort_values(['targets','labels']).reset_index(drop = True)
        
        # plot the RDMs
        RDM = distance.squareform(distance.pdist(features_average - features_average.mean(1).reshape(-1,1),
                                                 metric = 'cosine'))
        np.fill_diagonal(RDM, np.nan)
        fig,ax = plt.subplots(figsize = (12,10),)
        im = ax.imshow(RDM,
                       origin = 'lower',
                       cmap = plt.cm.coolwarm,
                       aspect = 'auto',
                       vmax = 1.2,
                       )
        plt.colorbar(im)
        ax.set(xticks = np.arange(features_average.shape[0]),
               yticks = np.arange(features_average.shape[0]),
               xticklabels = df_average['labels'],
               yticklabels = df_average['labels'],
               title = f'RDM of hidden layer - {hidden_size}\nmodel name: {model_name.upper()}')
        ax.set_xticklabels(df_average['labels'],
                           rotation = 90,
                           ha = 'center')
        
        # save things
        df_features.to_csv(os.path.join(cv_dir,f'{model_name}.csv'),index = False)
        fig.savefig(os.path.join(figure_dir,f'{model_name}.jpg'),
                    dpi = 300,
                    bbox_inches = 'tight')
        fig.savefig(os.path.join(figure_dir,f'{model_name}(light).jpg'),
                    # dpi = 300,
                    bbox_inches = 'tight')
        plt.close('all')