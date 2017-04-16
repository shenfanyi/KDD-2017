import pandas as pd
import datetime

new_tollgate1_0 = pd.read_csv('new_tollgate1_0.csv')
#print new_tollgate1_0.shape
#print new_tollgate1_0.iloc[-1]
#print new_tollgate1_0.loc[new_tollgate1_0['Wednesday']==1].count()
#print new_tollgate1_0['time_window']
#print new_tollgate1_0.iloc[0:2]

pieces1 = [new_tollgate1_0.iloc[0:24], new_tollgate1_0.iloc[178:180], new_tollgate1_0.iloc[24:58]]
data_1week = pd.concat(pieces1)
print data_1week.shape
#print data_1week

#print data_1week.values
data1 = pd.DataFrame(data_1week.values, index=range(0,60), columns=data_1week.columns)
#print data1

a = pd.DataFrame([range(1,61)])
b = a.T
b.columns=['timeid']
#print b

pieces2 = [data1.loc[:,'pressure':'precipitation'], b, data1['volume']]
data2 = pd.concat(pieces2, axis = 1)
print data2.shape
#print data2

data2.to_csv('oneweek1_0.csv')

print data2.iloc[:,0:8].shape



    


