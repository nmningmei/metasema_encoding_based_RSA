#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 03:44:52 2020

@author: nmei

This script to convert pickle data to numpy array and pandas dataframe,
so that the data can be read in python 3+

"""

import os
import pickle
import numpy as np
import pandas as pd
from glob import glob

subjects = os.listdir('../data/SearchlightStacked')
for sub in subjects:
    working_dir = '/export/home/nmei/public/Consciousness/metasema/Searchlight/{}'.format(sub)
    working_data = glob(os.path.join(working_dir,'*','*.pkl'))
    output_dir = '../data/Searchlight/{}'.format(sub)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    whole_brain_bold_stacked = []
    df_events = []
    for f in working_data:
        sub = f.split('/')[-2]
        roi_name = f.split('/')[-1].split('.')[0]
        data = pickle.load(open(f,'rb'))
        BOLD = data.samples.astype('float32')
        df = {}
        for key in data.sa.keys():
            df[key] = data.sa[key].value
        df = pd.DataFrame(df)
        
        whole_brain_bold_stacked.append(BOLD)
        df_events.append(df)
    whole_brain_bold_stacked = np.concatenate(whole_brain_bold_stacked)
    df_events = pd.concat(df_events)
    # save the data
    np.save(os.path.join(output_dir,'{}.npy'.format('while_brain_bold_stacked')),whole_brain_bold_stacked)
    df_events.to_csv(os.path.join(output_dir,'{}.csv'.format('while_brain_bold_stacked')),index = False)


























