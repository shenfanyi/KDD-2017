
import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


###  vol

##  look & handle data

# vol = pd.read_csv('training_20min_avg_volume.csv')
# # print vol.iloc[0]
# # print vol.shape

# totime = lambda x : datetime.strptime(x.split(',')[0], '[%Y-%m-%d %H:%M:%S')
# vol.loc[:,'time'] = vol.time_window.apply(totime)
# vol.loc[:,'date'] = vol.time.apply(datetime.date)
# vol.loc[:,'hour'] = vol.time.apply(datetime.time)
# vol.loc[:,'weekday'] = vol.time.apply(datetime.weekday)

# vol.loc[:,'nummin'] = [0]*len(vol)
# # print vol.loc[:,'nummin']
# for i in range(len(vol)):
# 	for j in vol.loc[:,'hour'].unique():
# 		if vol.loc[i,'hour'] == j:
# 			vol.loc[i,'nummin'] = list(vol.loc[:,'hour'].unique()).index(j) + 1
# 	# break

# vol.to_csv('volume_1.csv')


##  visualize

# vol = pd.read_csv('volume_1.csv')	

# vol = vol.groupby(by = ['tollgate_id','direction'])

# vol1 = vol.get_group((1,0))[~vol.get_group((1,0)).date.isin([
# 	'2016-09-21','2016-09-28','2016-09-29','2016-09-30','2016-10-01',
# 	'2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06','2016-10-07'
# 	])]

# # sns.lmplot( x='nummin', y='volume', data=vol1, hue='weekday', fit_reg=False)
# # sns.factorplot( x='nummin', y='volume', data=vol1, hue='weekday')
# sns.tsplot(time='nummin', value='volume',unit='date', 
# 	condition='weekday',data=vol1,
# 	err_style='unit_traces')
# plt.show()

# # print vol.get_group((1,0)).date.unique()




###  time

##  look & handle data

# tratime = pd.read_csv('training_20min_avg_travel_time.csv')
# # print time.iloc[0]
# # print time.shape

# totime = lambda x : datetime.strptime(x.split(',')[0], '[%Y-%m-%d %H:%M:%S')
# tratime.loc[:,'time'] = tratime.time_window.apply(totime)
# tratime.loc[:,'date'] = tratime.time.apply(datetime.date)
# tratime.loc[:,'hour'] = tratime.time.apply(datetime.time)
# tratime.loc[:,'weekday'] = tratime.time.apply(datetime.weekday)

# tratime.loc[:,'nummin'] = [0]*len(tratime)
# # print tratime.loc[:,'nummin']
# for i in range(len(tratime)):
# 	for j in tratime.loc[:,'hour'].unique():
# 		if tratime.loc[i,'hour'] == j:
# 			tratime.loc[i,'nummin'] = list(tratime.loc[:,'hour'].unique()).index(j) + 1
# 	# break

# tratime.to_csv('time_1.csv')


#  visualize

tratime = pd.read_csv('time_1.csv')	

tratime = tratime.groupby(by = ['tollgate_id','intersection_id'])

# tratime1 = tratime.get_group((3,'B'))[~tratime.get_group((3,'B')).date.isin([
# 	'2016-09-21','2016-09-28','2016-09-29','2016-09-30','2016-10-01',
# 	'2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06','2016-10-07'
# 	])]
tratime1 = tratime.get_group((3,'B'))

# sns.lmplot( x='nummin', y='avg_travel_time', data='tratime1', hue='weekday', fit_reg=False)
# sns.factorplot( x='nummin', y='avg_travel_time', data='tratime1', hue='weekday')
sns.tsplot(time='nummin', value='avg_travel_time',unit='date', 
	condition='date',data=tratime1,
	err_style='unit_traces')
plt.show()

# print tratime[0:2]