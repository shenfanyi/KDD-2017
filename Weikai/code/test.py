#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:44:16 2017

@author: carrey
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def str_to_dtime(string): #将时间字符串转化为时间类型
    dtime = datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    return dtime


def get_time_windows(): #获取窗口时间列表
    time_windows = []
    start_time = str_to_dtime('2016-10-18 06:00:00')  #第一个窗口时间起始点
    delta_hours = timedelta(hours = 9)
    delta_minutes = timedelta(minutes = 20)
    for i in range(7):
        for j in range(2):
            start_time += delta_hours * j
            for k in range(6):
                ltime = start_time + delta_minutes * k
                rtime = ltime + delta_minutes
                time_windows.append([ltime, rtime])
        start_time -= delta_hours
        start_time += timedelta(days = 1)
    return time_windows


def read_test(volume_file, travle_time_file): #读取测试数据文件，并转为DF类型
    volume_test = pd.read_csv(volume_file)
    volume_test['time'] = volume_test['time'].apply(str_to_dtime)
    
    travle_time_test = pd.read_csv(travle_time_file)
    travle_time_test['starting_time'] = travle_time_test['starting_time'].apply(str_to_dtime)
    return volume_test, travle_time_test
    

def transform_test(volume_test, travle_time_test): #将测试数据转为官方要求的格式
    time_windows = get_time_windows() #窗口时间列表
    volume_values = []  #流量数据存放列表
    travle_time_values = []   #平均通行时间数据存放列表
    str_time_windows = []  #存放窗口时间列表的字符串列表
    
    for w in time_windows:
        # 根据收费口和方向分组，并计算在窗口时间内的流量
        w_volume_test = volume_test.query('time >= @w[0] and time < @w[1]')
        group_volume_test = w_volume_test.groupby(['tollgate_id','direction'])['time'].count()
        
        # 根据路口和收费口分组，并计算在窗口时间内的平均通过时间
        w_ttime_test = travle_time_test.query('starting_time >= @w[0] and starting_time < @w[1]')
        group_travle_time_test = w_ttime_test.groupby(['intersection_id',
                                'tollgate_id'])['travel_time'].mean()
    
        # 将窗口时间列表存储为字符串形式
        str_time_window = '[' + datetime.strftime(w[0],"%Y-%m-%d %H:%M:%S") + \
            ',' + datetime.strftime(w[1],"%Y-%m-%d %H:%M:%S") + ')'
    
        for i in [1,2,3]:
            for j in [0,1]:
                try:
                    volume = group_volume_test.loc[np.s_[i,j],]
                    volume_values.append([i,str_time_window,j,volume])
                except KeyError:
                    continue
                
        for m in ['A','B','C']:
            for n in [1,2,3]:
                try:
                    avg_time = group_travle_time_test.loc[np.s_[m,n],]
                    travle_time_values.append([m,n,str_time_window,avg_time])
                except KeyError:
                    continue
                
        str_time_windows.append(str_time_window)
        
    trans_volume_test = pd.DataFrame(volume_values, 
                        columns = ['tollgate_id','time_window','direction','volumes'])
    
    trans_avg_time_test = pd.DataFrame(travle_time_values, 
                        columns = ['intersection_id','tollgate_id','time_window','avg_travle_time'])
    
    # 将得到的测试数据进行排序
    trans_volume_test.sort(columns = ['tollgate_id','time_window','direction'], inplace = True)
    trans_avg_time_test.sort(columns = ['intersection_id','tollgate_id','time_window'], inplace = True)

    return trans_volume_test, trans_avg_time_test,str_time_windows
    

def score(volume_pred, avg_time_pred): # 计算得分
    # 获取测试数据
    volume_test,travle_time_test = read_test('volume(table 6)_test1.csv',
                                         'trajectories(table 5)_test1.csv')
    volume, travle_time, time_windows = transform_test(volume_test, travle_time_test)
    
    # 将预测数据进行排序，保持与测试数据排序相同
    volume_pred.sort(columns = ['tollgate_id','time_window','direction'], inplace = True)
    avg_time_pred.sort(columns = ['intersection_id','tollgate_id','time_window'], inplace = True)
    
    # 计算误差列表
    volume_diff = abs((volume['volumes'] - volume_pred['volumes'])/volume['volumes'])
    travle_time_diff = abs((travle_time['avg_travle_time'] - avg_time_pred['avg_travle_time'])/travle_time['avg_travle_time'])
    
    # 计算误差和
    sum_volume_diff = volume_diff.sum()
    sum_travle_time_diff = travle_time_diff.sum()
    
    # 获取T值
    t = len(time_windows)
    
    # 获取得分
    score_volume = sum_volume_diff/(t * 5)
    score_travle_time = sum_travle_time_diff/(t * 6)
    
    return score_volume, score_travle_time
    



