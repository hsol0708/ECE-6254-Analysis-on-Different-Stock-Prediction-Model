import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

import pickle
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#plt.style.use('fivethirtyeight')
import xgboost as xgb
import sklearn
import tqdm
import random
# import janestreet
import tensorflow as tf
import datatable


SEED=1111
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
train = datatable.fread('../input/jane-street-market-prediction/train.csv').to_pandas()
train.shape
train.head(50)

#train = train.drop(['feature_113','feature_89','feature_101'], 1)

train = train.query('date > 85').reset_index(drop = True) 
train = train[train['weight'] != 0]

#train.fillna(train.mean(),inplace=True)

train['action'] = ((train['resp'].values) > 0).astype(int)


features = [c for c in train.columns if "feature" in c]
train.fillna(train.mean(),inplace=True)   
features.remove('feature_0')
len(features)
train.shape

train['resp'] = (((train['resp'].values)*train['weight']) > 0).astype(int)
train['resp_1'] = (((train['resp_1'].values)*train['weight']) > 0).astype(int)
train['resp_2'] = (((train['resp_2'].values)*train['weight']) > 0).astype(int)
train['resp_3'] = (((train['resp_3'].values)*train['weight']) > 0).astype(int)
train['resp_4'] = (((train['resp_4'].values)*train['weight']) > 0).astype(int)

f_mean = np.mean(train[features[1:]].values,axis=0)

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp']

#X_train = train.loc[:, train.columns.str.contains('feature')]

len(features)
X_train=train[features].values
#y_train = (train.loc[:, 'action'])

y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
import optuna.integration.lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# modeling step 
params={"num_leaves":300,
       "max_bin":450,
       "feature_fraction":0.52,
       "bagging_fraction":0.52,
       "objective":"binary",
       "learning_rate":0.05,
       "boosting_type":"gbdt",
       "metric":"auc"
       }
models = [] # list of model , we will train 
for i in range(y_train.shape[1]):
    xtr,xval,ytr,yval = train_test_split(X_train ,y_train[:,i],test_size=0.2,stratify=y_train[:,i])
   
    d_train = lgbm.Dataset(xtr,label=ytr)
    d_eval = lgbm.Dataset(xval,label=yval,reference=d_train)
    clf = lgbm.train(params,d_train,valid_sets=[d_train,d_eval],num_boost_round=1000,\
                    early_stopping_rounds=50,verbose_eval=50)
    models.append(clf)

    VER = 1
model_name = 'lgb_model_'+str(VER)+'.bin'
pickle.dump(models, open(model_name, 'wb'))
fig,ax = plt.subplots(figsize=(25,50))
lgbm.plot_importance(clf, ax=ax,importance_type='gain',max_num_features=130)
plt.show()


#preds = clf.predict(xtr)
#pred_labels = np.rint(preds)


    
#accuracy = sklearn.metrics.accuracy_score(ytr, pred_labels)