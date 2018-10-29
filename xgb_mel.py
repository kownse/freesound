import os, random, math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb

import librosa
import librosa.display

from scipy.stats import skew, kurtosis
from sklearn.cross_validation import StratifiedKFold
from prettytable import PrettyTable
from tqdm import tqdm_notebook, tqdm_pandas
tqdm_notebook().pandas(smoothing=0.7)

import IPython
import IPython.display as ipd

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import kaggle_util
from util import *
from xgboost import XGBClassifier

DEBUG = 0
nfold = 10
nround = 50000
if DEBUG:
    nfold = 2
    nround = 5
  
nrows = None if not DEBUG else 1000

if __name__ == "__main__":
    train = kaggle_util.reduce_mem_usage(pd.read_csv('../data/train_mel.csv', nrows=nrows))
    test = kaggle_util.reduce_mem_usage(pd.read_csv('../data/test_mel.csv', nrows=nrows))
    #y = pd.get_dummies(train.label)
    y = np.load('../cache/train_y.npy')
    if DEBUG:
        y = y[:nrows]

    LABELS = list(train.label.unique())
    n_categories = len(LABELS)
    
    train = train.drop(['fname', 'label', 'manually_verified'], axis=1)
    feature_names = list(test.drop(['fname', 'label'], axis=1).columns.values)
    test = test.drop(['fname', 'label'], axis=1).values

    #labels = y.columns.values
    #y_label = y.values
    y_label = [np.argmax(row) for row in y]
    
    PREDICTION_FOLDER = '../result/predictions/xgb'
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)

    cvscores = []
    skf = StratifiedKFold(y_label, n_folds=nfold)
    for i, (train_split, val_split) in enumerate(skf):
        X_train = train.iloc[train_split].values
        y_train = [np.argmax(row) for row in y[train_split]]
        X_valid = train.iloc[val_split].values
        y_valid = [np.argmax(row) for row in y[val_split]] 

        print(X_train.shape, X_valid.shape)
        #exit()
        
        clf = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=nround,
                        n_jobs=4, random_state=0, reg_alpha=0.2, 
                        colsample_bylevel=0.9, colsample_bytree=0.9)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='mlogloss', 
                verbose=True,
                early_stopping_rounds = 200,
               )

        p = clf.predict_proba(X_valid)
        valid_score = get_valid_score(y_valid, p)
        print("Score = {:.4f}".format(valid_score))
        cvscores.append(valid_score)
        
        pre_test = clf.predict_proba(test, clf.best_iteration)
        savepath = "/p{}.npy"
        savepath = savepath.format(i)
        np.save(PREDICTION_FOLDER + savepath, pre_test)

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names)

    cvmean = np.mean(cvscores)
    cvstd = np.std(cvscores)
    print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
    actual_prefix = '{:.2f}_{:.2f}'.format(cvmean, cvstd)
    ensemble(LABELS, nfold, [PREDICTION_FOLDER], actual_prefix, 'xgb', False)