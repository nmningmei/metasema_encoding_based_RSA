#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 08:49:35 2021

@author: nmei
"""
import os,gc

import numpy  as np
import pandas as pd

from scipy.spatial import distance
from scipy.stats   import spearmanr
from joblib        import Parallel,delayed
try:
    from nipype.interfaces import fsl
except Exception as e:
    print(str(e))

def groupby_average(list_of_arrays,df,groupby = ['trials'],axis = 0):
    """
    To compute the average groupby the dataframe
    list_of_arrays: list of ndarrays
    df: pandas.DataFrame,
        dataframe object
    groupby: str or list of strings
    """
    place_holder = {ii:[] for ii in range(len(list_of_arrays))}
    for ii,array in enumerate(list_of_arrays):
        place_holder[ii] = np.array([np.mean(array[df_sub.index],axis) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([df_sub.iloc[0,:].to_frame().T for ii,df_sub in df.groupby(groupby)])
    return place_holder,df_average

def feature_normalize(features):
    """
    Normalize individual row of the matrix
    """
    features = features - features.mean(1).reshape(-1,1)
    return features

def searchlight_function_unit(sphere_bold_singals, mask, myrad, broadcast_variable):
    """
    sphere_bold_singals: BOLD
    mask: mask array
    myrad: not use
    broadcast_variable: label -- features RDM
    """
    BOLD    = sphere_bold_singals[0][mask,:].T.copy()
    RDM_y   = broadcast_variable.copy()
#    print(BOLD.shape)
    # pearson correlation
    RDM_X   = distance.pdist(feature_normalize(BOLD),'correlation')
    D,p     = spearmanr(RDM_X ,RDM_y)
#    print(D)
    return D

def convert_individual_space_to_standard_space(brain_map_individual_space,
                                               standard_brain,
                                               in_transformation_matrix_file,
                                               out_transformation_matrix_file,
                                               brain_map_standard_space,
                                               run_algorithm = False,
                                               ):
    """
    FSL FLIRT alogirthm called by nipype
    
    brain_map_individual_space: string or os.path.abspath
        the path of the brain map in individual space
    standard_brain: string or os.path.abspath
        the path of the MNI standard brain 2mm brain map
    in_transformation_matrix_file: string or os.path.abspath
        the path of the transformation matrix that convert individual space to standard space
    out_transformation_matrix_file: string or os.path.abspath
        an output transformation matrix
    brain_map_standard_space: string or os.path.abspath
        output file path
    run_algorithm: bool, default = False
    """
    
    
    flt = fsl.FLIRT()
    flt.inputs.in_file          = os.path.abspath(brain_map_individual_space)
    flt.inputs.reference        = os.path.abspath(standard_brain)
    flt.inputs.in_matrix_file   = os.path.abspath(in_transformation_matrix_file)
    flt.inputs.out_matrix_file  = os.path.abspath(out_transformation_matrix_file)
    flt.inputs.out_file         = os.path.abspath(brain_map_standard_space)
    flt.inputs.output_type      = 'NIFTI_GZ'
    flt.inputs.apply_xfm        = True
    if run_algorithm:
        flt.run()
    else:
        print(flt.cmdline)
    return flt

def nipype_fsl_randomise(input_file,
                         mask_file,
                         base_name,
                         tfce                   = True,
                         var_smooth             = 6,
                         demean                 = False,
                         one_sample_group_mean  = True,
                         n_permutation          = int(1e4),
                         quiet                  = False,
                         run_algorithm          = False,
                         ):
    """
    Run FSL-randomise for a one-tailed one-sample t test, correct
        by TFCE
    
    input_file: string or os.path.abspath
        4DNifti1Image
    mask_file: string or os.path.abspath
        3DNifti1Image
    base_name: string or os.path.abspath
        base name
    tfce: bool, default = True
        to correct the p values with TFCE
    var_smooth: int, default = 6
        size for variance smoothing, unit = mm
    demean: bool, default = False
        temporally remove the mean before computation
    one_sample_group_mean: bool, default = True
        one-sample t test
    n_permutation: int, default = 10000
        number of permutations, set to 0 for exhausive
    quiet: bool, default = False
        set to True to surpress the outputs
    run_algorithm: bool, default = False
        run the algorithm or print the command line code
    """
    rand                        = fsl.Randomise()
    if quiet:
        rand.inputs.args        = '--quiet'
    rand.inputs.in_file         = os.path.abspath(input_file)
    rand.inputs.mask            = os.path.abspath(mask_file)
    rand.inputs.tfce            = tfce
    rand.inputs.var_smooth      = var_smooth
    rand.inputs.base_name       = os.path.abspath(base_name)
    rand.inputs.demean          = demean
    rand.inputs.one_sample_group_mean = one_sample_group_mean
    rand.inputs.num_perm        = n_permutation
    rand.inputs.seed            = 12345
    
    if run_algorithm:
        rand.run()
    else:
        print(rand.cmdline)
    return rand

def load_event_files(f,idx = 3):
    """
    f: string or os.path
    idx: int, default = 3
    """
    temp = f.split('/')
    df          = pd.read_csv(f)
    df['sub']   = temp[idx]
    return df

def load_computational_features(results_dir,model_name):
    """
    results_dir: string or os.path
        It is for bash script generating
    model_name: string
        It is for the dictionary defined below
    """
    directories = {'vgg19':     f'{results_dir}/computer_vision_features_no_background_caltech',
                   'mobilenet': f'{results_dir}/computer_vision_features_no_background_caltech',
                   'resnet50':  f'{results_dir}/computer_vision_features_no_background_caltech',
                   'fasttext':  f'{results_dir}/word2vec_features',
                   'glove':     f'{results_dir}/word2vec_features',
                   'word2vec':  f'{results_dir}/word2vec_features',}
    filename = os.path.join(directories[model_name],f'{model_name}.csv')
    df = pd.read_csv(filename)
    return df

def define_label_map(one_hot = True):
    if one_hot:
        return {'animal':[0,1],
                'tool'  :[1,0]}
    else:
        return {'animal':0,
                'tool'  :1}

def load_data_for_randomise(working_data,
                            folder_name,
                            masker,
                            return_tanh = True,):
    # mask the output before we conduct statistical inference
    if folder_name == 'encoding':
        data                = []
        for item in working_data:
            try:
                data.append(masker.transform(item)[0])
                
            except:
                pass
        data                = np.array(data)
        data[0 > data]      = 0
        data_tanh = data.copy()
    else:
        gc.collect()
        data = Parallel(n_jobs = -1, verbose = 1)(delayed(masker.transform)(**{
            'imgs':item}) for item in working_data)
        gc.collect()
        # convert data to z scores
        if return_tanh:
            data_tanh       = np.arctanh(data)
        else:
            data_tanh       = data.copy()
    return data_tanh