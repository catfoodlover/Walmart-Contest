# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:32:54 2015

@author: home
About this data set:
Fineline Number: a more refined category for each of the products(items commonly
purchased together share a fineline number)
ScanCount: the number of a specific item purchased in a visit, negatives are
returned items
"""
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import scipy
from sklearn import preprocessing
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer



walmart = pd.read_csv('~/Downloads/train 3.csv')
testset = pd.read_csv('~/Downloads/test 3.csv')

#here I copy it in case I do something to my data set
walltest = walmart.copy()

#convert the NaNs to -1000
grpwall = walltest.fillna(-1000)
testfix = testset.fillna(-1000)

'''this part is tricky, to join your new feature to the existing dataframe
you need to make sure you do the join correctly.  You need to tell python to join
on the right index because the pd.groupby doesn't give you column names to join on'''

Returned = grpwall.merge(pd.DataFrame({'ItemReturned':grpwall.ScanCount <= 0}),\
 left_index = True, right_index = True)

Returned['RealSC'] = Returned.ScanCount
Returned['RealSC'].ix[Returned.ItemReturned]=0

containsReturned = Returned.merge(pd.DataFrame\
({'ContainsReturned':Returned.groupby('VisitNumber').sum().ItemReturned>=1}),\
 left_on=['VisitNumber'], right_index=True)

NumberReturned = containsReturned.merge(pd.DataFrame\
({'NumReturned':containsReturned.groupby('VisitNumber').sum().ItemReturned}),\
left_on=['VisitNumber'], right_index=True)
 
nUnqItems = NumberReturned.merge(pd.DataFrame\
({'itemcount':NumberReturned.groupby('VisitNumber').size()}),\
 left_on=['VisitNumber'], right_index=True)

addFeats = nUnqItems.merge(pd.DataFrame\
({'totalcount':nUnqItems.groupby('VisitNumber').sum().RealSC}),\
 left_on=['VisitNumber'], right_index=True)
 
grpUpc = pd.DataFrame({'percentTrip':addFeats.groupby\
(['VisitNumber','Upc']).size()/addFeats.groupby(['VisitNumber']).size()})
prcntTrip = addFeats.merge(grpUpc, left_on=['VisitNumber', 'Upc'], right_index = True)

grpLineItem = pd.DataFrame({'percentLineTrip':addFeats.groupby\
(['VisitNumber','FinelineNumber']).size()/addFeats.groupby(['VisitNumber']).size()})
prcntTripII = prcntTrip.merge(grpLineItem, left_on=['VisitNumber', 'FinelineNumber'], right_index = True)

grpDepartItem = pd.DataFrame({'percentDepartTrip':addFeats.groupby\
(['VisitNumber','DepartmentDescription']).size()/addFeats.groupby(['VisitNumber']).size()})
prcntTripIII = prcntTripII.merge(grpDepartItem,\
 left_on=['VisitNumber', 'DepartmentDescription'], right_index = True)


 
#Here is where I create the same features in the test data set
Returnedtest = testfix.merge(pd.DataFrame({'ItemReturned':testfix.ScanCount <= 0}),\
 left_index = True, right_index = True)

Returnedtest['RealSC'] = Returnedtest.ScanCount
Returnedtest['RealSC'].ix[Returnedtest.ItemReturned]=0
 
containsReturnedtest = Returnedtest.merge(pd.DataFrame\
({'ContainsReturned':Returnedtest.groupby('VisitNumber').sum().ItemReturned>=1}),\
 left_on=['VisitNumber'], right_index=True)

NumberReturnedtest = containsReturnedtest.merge(pd.DataFrame\
({'NumReturned':containsReturnedtest.groupby('VisitNumber').sum().ItemReturned}),\
left_on=['VisitNumber'], right_index=True)

nUnqItemtest = NumberReturnedtest.merge(pd.DataFrame\
({'itemcount':NumberReturnedtest.groupby('VisitNumber').size()}),\
left_on=['VisitNumber'], right_index=True)

addFeatest = nUnqItemtest.merge(pd.DataFrame\
({'totalcount':nUnqItemtest.groupby('VisitNumber').sum().RealSC}),\
 left_on=['VisitNumber'], right_index=True)

grpUpctest = pd.DataFrame({'percentTrip':addFeatest.groupby\
(['VisitNumber','Upc']).size()/addFeatest.groupby(['VisitNumber']).size()})
prcntTriptest = addFeatest.merge(grpUpctest, left_on=['VisitNumber', 'Upc'], right_index = True)

grpLineItemT = pd.DataFrame({'percentLineTrip':addFeatest.groupby\
(['VisitNumber','FinelineNumber']).size()/addFeatest.groupby(['VisitNumber']).size()})
prcntTriptestII = prcntTriptest.merge(grpLineItemT, left_on=['VisitNumber', 'FinelineNumber'], right_index = True)

grpDepartItemT = pd.DataFrame({'percentDepartTrip':addFeatest.groupby\
(['VisitNumber','DepartmentDescription']).size()/addFeatest.groupby(['VisitNumber']).size()})
prcntTriptestIII = prcntTriptestII.merge(grpDepartItemT, left_on=['VisitNumber', 'DepartmentDescription'], right_index = True)


#Here is where I split off the Y(Triptype) and visit number in the training set
wal_total_x = prcntTripIII.iloc[:,2:]
wal_total_y = prcntTripIII.iloc[:,0]

#Here is where I remove the Visit Number from the test set
wal_test_x = prcntTriptestIII.iloc[:,1:]

#Here I check their shape for a sanity check
np.shape(wal_total_x)
np.shape(wal_test_x)

#Here I convert the training and test sets to transposed dictionaries
#and then transform them into sparse matrix
train = wal_total_x.T.to_dict().values()
test = wal_test_x.T.to_dict().values()
vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

#Here I use labe encoder to convert the labels of a range of values between
#0-37, the training and test sets are converted to sparse matrices
le = preprocessing.LabelEncoder()
le.fit(wal_total_y)
train_label = le.transform(wal_total_y)

dtrain = xgb.DMatrix(train[20000:, :], label = train_label[20000:].astype(float))
dval = xgb.DMatrix(train[:20000, :], label = train_label[:20000].astype(float))
dtest = xgb.DMatrix(test)

#here is where I actually run xgboost and all the parameters are set
param = {'bst:max_depth':13,'colsample_bytree': 0.65, 'bst:eta':0.375,'colsample_bytree': 0.65,\
  'silent':1, 'objective':'multi:softprob', "num_class": 38,'gamma': 0.95,'subsample': 0.95 }
param['eval_metric'] = 'mlogloss'
evallist  = [(dval,'eval'), (dtrain,'train')]
plst = param.items()
num_round = 50
bst = xgb.train(plst, dtrain, num_round, evallist )
ypred = bst.predict(dtest)
print ypred


'''Here is where I make my submission,  I convert the output into a dataframe.  I then add the index and column names'''
#submission = randomForest.predict_proba(ypred)
#submission = randomForest.predict_proba(keeperstest)
submissiondf = pd.DataFrame(ypred)
dex = testset.iloc[:,0]
submurge = pd.concat([dex,submissiondf], axis = 1)
avgmurg = submurge.groupby(submurge.VisitNumber).mean()
avgmurg.columns = ['TripType_3','TripType_4','TripType_5','TripType_6','TripType_7',\
'TripType_8','TripType_9','TripType_12','TripType_14','TripType_15','TripType_18',\
'TripType_19','TripType_20','TripType_21','TripType_22','TripType_23','TripType_24',\
'TripType_25','TripType_26','TripType_27','TripType_28','TripType_29','TripType_30',\
'TripType_31','TripType_32','TripType_33','TripType_34','TripType_35','TripType_36',\
'TripType_37','TripType_38','TripType_39','TripType_40','TripType_41','TripType_42',\
'TripType_43','TripType_44','TripType_999']
len(avgmurg.columns)
#here is where I export my submission
avgmurg.to_csv('Desktop/KaggleSub_15.csv')


