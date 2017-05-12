

import pandas as pd
import numpy as np

am = pd.read_csv('morning_data.csv')
# print am.iloc[0:1]

am['date'] = pd.Series(np.random.randn(len(am)), index=am.index)
print am

for i in range(len(am.time_window)):
	am.time_window[i] =  am.time_window[i].split(' ')
	am['date'][i] = am.time_window[i][0]
	am.time_window[i] = am.time_window[i][2]
	

# z = am.time_window[0]
# print z
# z = z.split(' ')
# print z[0]

print am.iloc[0:1]
am.to_csv('am.csv')




pm = pd.read_csv('afternoon_data.csv')
# print am.iloc[0:1]

pm['date'] = pd.Series(np.random.randn(len(pm)), index=pm.index)
print pm

for i in range(len(pm.time_window)):
	pm.time_window[i] =  pm.time_window[i].split(' ')
	pm['date'][i] = pm.time_window[i][0]
	pm.time_window[i] = pm.time_window[i][2]

print pm.iloc[0:1]
pm.to_csv('pm.csv')