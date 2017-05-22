
import pandas as pd

train = pd.read_csv('trainset_task2_version2.csv')
test = pd.read_csv('testset_task2_version2.csv')

print train.iloc[0]
# print test.iloc[0:5]
print train.shape
print test.shape

# train10 = train[train.eval('tollgate_id == 1 & direction == 0')]
# train11 = train[train.eval('tollgate_id == 1 & direction == 1')]
# train20 = train[train.eval('tollgate_id == 1 & direction == 0')]
# train30 = train[train.eval('tollgate_id == 3 & direction == 0')]
# train31 = train[train.eval('tollgate_id == 3 & direction == 1')]
# print train10.iloc[0:5]
# print train10.shape
# print train11.shape
# print train20.shape
# print train30.shape
# print train31.shape

# test10 = test[test.eval('tollgate_id == 1 & direction == 0')]
# test11 = test[test.eval('tollgate_id == 1 & direction == 1')]
# test20 = test[test.eval('tollgate_id == 1 & direction == 0')]
# test30 = test[test.eval('tollgate_id == 3 & direction == 0')]
# test31 = test[test.eval('tollgate_id == 3 & direction == 1')]
