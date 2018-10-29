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
from dataset import *

DEBUG = 0
nfold = 5
nround = 50000
if DEBUG:
    nfold = 2
    nround = 5
  
nrows = None if not DEBUG else 1000

def statistic_features():
    argucnt = 5
    train = kaggle_util.reduce_mem_usage(pd.read_csv('../data/train_mel.csv', nrows=nrows))
    test = kaggle_util.reduce_mem_usage(pd.read_csv('../data/test_mel.csv', nrows=nrows))
    #y = pd.get_dummies(train.label)
    y_label = np.load('../cache/train_y.npy')
    if DEBUG:
        y_label = y_label[:nrows]

    LABELS = list(train.label.unique())
    n_categories = len(LABELS)
    
    train = train.drop(['fname', 'label', 'manually_verified'], axis=1)
    feature_names = list(test.drop(['fname', 'label'], axis=1).columns.values)
    test = test.drop(['fname', 'label'], axis=1)
    
    argucnt = 2
    mixup = 3
    mixup_alpha = 1
    train = train[:len(y_label) * argucnt]
    
    y = np.copy(y_label)
    for i in range(argucnt):
        y = np.vstack([y, y_label])
    
    y_label = [np.argmax(row) for row in y_label]
    len_y = len(y_label)
    print(len_y, train.shape)
    #exit()
    PREDICTION_FOLDER = '../result/predictions/lgb'
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)

    
    cvscores = []
    skf = StratifiedKFold(y_label, n_folds=nfold)
    for i, (train_split, val_split) in enumerate(skf):
        train_split = np.hstack([(train_split + len_y * i) for i in range(argucnt)])
        X_train = train.iloc[train_split].values
        y_train = y[train_split]
        #y_train = [np.argmax(row) for row in y[train_split]]

        if mixup > 0:
            x_train_sub = None
            y_train_sub = None
            for j in range(mixup):
                print('mixup', mixup)
                x_tmp, y_tmp = mixup_all(X_train, y_train, mixup_alpha)
                x_train_sub = x_tmp if x_train_sub is None else np.vstack((x_train_sub, x_tmp))
                y_train_sub = y_tmp if y_train_sub is None else np.vstack((y_train_sub, y_tmp))
            X_train = x_train_sub
            y_train = y_train_sub

        y_train = [np.argmax(row) for row in y_train]
        
        #exit()
        X_valid = train.iloc[val_split].values
        #y_valid = y[val_split]
        y_valid = [np.argmax(row) for row in y[val_split]] 
        
        print(X_train.shape, X_valid.shape)
        #print(feature_names)

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'max_depth': 5,
            'num_leaves': 9,
            'learning_rate': 0.005,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 2,
            'num_threads': 8,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0,
            'num_class': n_categories
        }

        clf = lgb.train(params, d_train, num_boost_round=nround, 
                        valid_sets=d_valid, verbose_eval=200, 
                        early_stopping_rounds=100)
        p = clf.predict(X_valid, num_iteration=clf.best_iteration)

        #predictions = [list(np.argsort(p[i])[::-1][:3]) for i in range(len(p))]
        #actual = [[i] for i in y_valid]
        #valid_score = mapk(actual, predictions, k=3)
        valid_score = get_valid_score(y_valid, p)
        print("Score = {:.4f}".format(valid_score))
        cvscores.append(valid_score)
        
        f, ax = plt.subplots(figsize=[7,100])
        lgb.plot_importance(clf, max_num_features=200, ax=ax)
        plt.title("Light GBM Feature Importance")
        plt.savefig('feature_import.png', bbox_inches='tight')

        pre_test = clf.predict(test, num_iteration=clf.best_iteration)
        savepath = "/p{}.npy"
        savepath = savepath.format(i)
        np.save(PREDICTION_FOLDER + savepath, pre_test)

    cvmean = np.mean(cvscores)
    cvstd = np.std(cvscores)
    print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
    actual_prefix = '{:.3f}'.format(cvmean)
    ensemble(LABELS, nfold, [PREDICTION_FOLDER], actual_prefix, 'lgb', False)

if __name__ == "__main__":
    
    statistic_features()