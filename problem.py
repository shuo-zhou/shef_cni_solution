# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:13:03 2019

@author: sz144
"""

import os
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import FunctionTransformer

#base_dir = 'D:/CNI19'
base_dir = '/home/shuoz/data/CNI19'

#train_path = 'D:/CNI19/2019_CNI_TrainingRelease-master'
#valid_path = 'D:/CNI19/2019_CNI_ValidationRelease-master'
#
#pheno_train = os.path.join(train_path, 'SupportingInfo/phenotypic_training.csv')
#pheno_valid = os.path.join(valid_path, 'SupportingInfo/phenotypic_validation.csv')

def _load_fmri(sub_ids, path, atlas='cc200'):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return [pd.read_csv(os.path.join(path,'%s/timeseries_%s.csv'%(sub_id, atlas)), 
                        header=None).values.T
                     for sub_id in sub_ids]


def _load_data(partition='Training', atlas='cc200'):
    path = os.path.join(base_dir, '2019_CNI_%sRelease-master'%partition)
    
    pheno_path = os.path.join(path, 'SupportingInfo/phenotypic_%s.csv'%partition.lower())
    
    pheno_df = pd.read_csv(pheno_path)
    
    data = _load_fmri(pheno_df['Subj'], os.path.join(path, partition), atlas=atlas)
    
    pheno_df_ = pheno_df.copy()
    
    for i in pheno_df_.index.values:
        if pheno_df_.loc[i,'DX'] == 'ADHD':
            pheno_df_.loc[i,'DX'] = 1
        else:
            pheno_df_.loc[i,'DX'] = -1
            
    return data, pheno_df_

def get_data(kind='tangent', atlas='aal'):
    pheno_path = os.path.join(base_dir, 'pheno.csv')
    
    data_path = os.path.join(base_dir, 'X_%s_%s.npy'%(atlas, kind))
    
    if os.path.exists(data_path):
        X = np.load(data_path)
    else:
        X_train, pheno_train = get_train_data(atlas=atlas)
        X_valid, pheno_valid = get_valid_data(atlas=atlas)
        X_all = X_train + X_valid
        measure = ConnectivityMeasure(kind=kind, vectorize=True)
        X = measure.fit_transform(X_all) 
        np.save(data_path, X)
        if not os.path.exists(pheno_path):
            pheno = pd.concat([pheno_train, pheno_valid])
            pheno.to_csv(pheno_path, index=False)
    pheno = pd.read_csv(pheno_path)
    return X, pheno
    

def get_train_data(atlas='cc200'):
    return _load_data(partition='Training', atlas=atlas)

def get_valid_data(atlas='cc200'):
    return _load_data(partition='Validation', atlas=atlas)
        




