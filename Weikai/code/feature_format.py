#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:50:26 2017

@author: carrey
"""
import pandas as pd

#def map_window(x):
#    t = pd.Series(x.unique())
#    a = pd.Series(range(len(t)), index = t.values)
#    map_a = x.map(a)
#    return map_a

def feature_format(task):
    if task == 1:
        pd_traj_train = pd.read_csv('../../Processed_data/by_weikai/big_train_trajectorise.csv')
    elif task == 2:
        pd_vol_train = pd.read_csv('../../Processed_data/by_weikai/big_train_volume.csv')
        pd_vol_test = pd.read_csv('../../Processed_data/by_weikai/big_test_volume.csv')
        pd_vol_train = pd_vol_train.set_index(['time'])
        pd_vol_test = pd_vol_test.set_index(['time'])
        vol_train = pd_vol_train.groupby(['time_window','tollgate_id','direction','weekday','hour'])\
                    .size().reset_index().rename(columns = {0:'volume'})
        vol_test = pd_vol_test.groupby(['time_window','tollgate_id','direction','weekday','hour'])\
                    .size().reset_index().rename(columns = {0:'volume'})
                    
        x = pd.Series(vol_train['time_window'].unique())
        s = pd.Series(range(len(x)),index = x.values)
        vol_train['window_n'] = vol_train['time_window'].map(s)
        vol_test['window_n'] = vol_test['time_window'].map(s)
#        print vol_test.tail()
        
        feature_train = vol_train.drop('volume', axis = 1).set_index(['time_window'])
        feature_test = vol_test.drop('volume',axis = 1).set_index(['time_window'])
        values_train = vol_train['volume'].values
        values_test = vol_test['volume'].values
        
    else:
        print "Error: The task should be 1 or 2."
        
    return feature_train, feature_test, values_train, values_test
        
feature_format(2)