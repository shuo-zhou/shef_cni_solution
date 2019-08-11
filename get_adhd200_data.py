#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:50:00 2019

@author: shuoz
"""
import os
import numpy as np
import pandas as pd

basedir = '/home/shuoz/data/adhd200/athena_TCs'

sites=['NYU','NeuroIMAGE','KKI','OHSU','Peking_1','Peking_2','Peking_3',
       'Pittsburgh','WashU']



def load_adhd200(atlas='cc200'):
    path = os.path.join(basedir,'ADHD200_%s'%atlas)
    tc_data = []
    pheno_list = []
    for site in sites:
        site_path=os.path.join(path, site)        
        site_pheno = pd.read_csv(os.path.join(site_path, '%s_phenotypic.csv'%site))
        for idx in site_pheno.index.values:
            if site_pheno.loc[idx, 'DX'] == 0:
                site_pheno.loc[idx, 'DX'] = -1
            else:
                site_pheno.loc[idx, 'DX'] = 1
        pheno_list.append(site_pheno)
        for sub_id in site_pheno['ScanDirID']:
            if len(str(sub_id))<7:
                sub_id_str = '00%s'%sub_id
            else:
                sub_id_str = str(sub_id)
            sub_path=os.path.join(site_path, sub_id_str)
            for session in range(1,5):
                sub_tc_path = os.path.join(sub_path,'sfnwmrda%s_session_%s_rest_1_%s_TCs.1D'%(sub_id_str,session,atlas))
                if os.path.exists(sub_tc_path):
                    sub_tc = pd.read_csv(sub_tc_path, sep='\t')
                    tc_data.append(sub_tc.iloc[:,2:].values)
                    break
    pheno = pd.concat(pheno_list, axis=0)
    return pheno, tc_data
                
            
        