#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:50:26 2017

@author: carrey
"""
import pandas as pd


def feature_format(task):
    if task == 1:
        pd_traj_train = pd.read_csv('../../Processed_data/by_weikai/big_train_trajectorise.csv')
    elif task == 2:
        pd_vol_train = pd.read_csv('../../Processed_data/by_weikai/big_train_volume.csv')
        print pd_vol_train.head()
        
        
    else:
        print "Error: The task should be 1 or 2."
        
feature_format(task = 2)