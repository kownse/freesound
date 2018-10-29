import os, random, math

import pandas as pd
import numpy as np
import lightgbm as lgb

import librosa
import librosa.display

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split

#from prettytable import PrettyTable
from tqdm import tqdm_notebook, tqdm_pandas
tqdm_notebook().pandas(smoothing=0.7)

import IPython
import IPython.display as ipd

import matplotlib as mpl
mpl.rcParams['font.size'] = 14

from util import Config, DataGenerator, audio_norm, getCacheTrainData, getCacheTestData

if __name__ == "__main__":
    sampling_rate=16000
    audio_duration=5
    
    nrows = None
    train = pd.read_csv("../data/train.csv", nrows=nrows)
    test = pd.read_csv("../data/sample_submission.csv", nrows=nrows)

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    
    getCacheTrainData(train, '../data/audio_train/', sampling_rate, audio_duration)
    getCacheTestData(test, '../data/audio_test/', sampling_rate, audio_duration)