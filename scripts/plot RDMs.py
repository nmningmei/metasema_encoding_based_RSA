#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:25:11 2021

@author: nmei
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from scipy.spatial import distance
from matplotlib import pyplot as plt

sns.set_context('poster')

if __name__ == "__main__":
    working_dir = '../results/*features*'
    working_data = glob(os.path.join(working_dir,'*.csv'))
    words = np.load('../data/unique_words/words.npy').astype(str)
    model_name_map      = {'vgg19':     'VGG19',
                           'mobilenet': 'MobileNetV2',
                           'resnet50':  'ResNet50',
                           'fasttext':  'Fast Text',
                           'glove':     'GloVe',
                           'word2vec':  'Word2Vec',}
    model_type_map      = {'vgg19':     'cv',
                           'mobilenet': 'cv',
                           'resnet50':  'cv',
                           'fasttext':  'wv',
                           'glove':     'wv',
                           'word2vec':  'wv',}
    output_dir          = {'vgg19':     '../figures/cv_features',
                           'mobilenet': '../figures/cv_features',
                           'resnet50':  '../figures/cv_features',
                           'fasttext':  '../figures/word2vec_features',
                           'glove':     '../figures/word2vec_features',
                           'word2vec':  '../figures/word2vec_features',}
    for filename in working_data:
        model_name = filename.split('/')[-1].split('.')[0]
        df_features = pd.read_csv(filename)
        features = np.array([df_features[word] for word in words])
        if model_type_map[model_name] == 'cv':
            features = -features
        RDM = distance.squareform(distance.pdist(features - features.mean(1).reshape(-1,1),'cosine'))
        
        np.fill_diagonal(RDM, np.nan)
        fig,ax = plt.subplots(figsize = (12,11),)
        im = ax.imshow(RDM,
                       origin = 'lower',
                       cmap = plt.cm.coolwarm,
                       aspect = 'auto',
                       vmax = .9,
                       )
        ax.axvline(words.shape[0] / 2 - 0.5, linestyle = '--',color = 'black',alpha = 1.)
        ax.axhline(words.shape[0] / 2 - 0.5, linestyle = '--',color = 'black',alpha = 1.)
        plt.colorbar(im)
        ax.set(xticks = np.arange(words.shape[0]),
               yticks = np.arange(words.shape[0]),
               xticklabels = words,
               yticklabels = words,
               title = f'RDM of hidden layer - 300\nmodel name: {model_name_map[model_name]}')
        ax.set_xticklabels(words,
                           rotation = 90,
                           ha = 'center')
        fig.savefig(os.path.join(output_dir[model_name],
                                 f'{model_name}.jpg'),
                    dpi = 300,
                    bbox_inches = 'tight')
        fig.savefig(os.path.join(output_dir[model_name],
                                 f'{model_name}(light).jpg'),
#                    dpi = 300,
                    bbox_inches = 'tight')
        plt.close('all')