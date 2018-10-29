import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../data"))
import librosa
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import skew
SAMPLE_RATE = 44100

#from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm, tqdm_pandas

tqdm.pandas()
import scipy
from util import *

def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

#returns mfcc features with mean and standard deviation along time
def get_mfcc(name, path):
    b, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
        ft2 = librosa.feature.zero_crossing_rate(b)[0]
        ft3 = librosa.feature.spectral_rolloff(b)[0]
        ft4 = librosa.feature.spectral_centroid(b)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.min(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc)))
    except:
        print('bad file')
        return pd.Series([0]*115)
    
def extract_features(files, path, flex):
    file_name = '../cache/mfcc_{}.csv'.format(flex)
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    
    features = {}

    cnt = 0
    for f in tqdm(files):
        features[f] = {}

        fs, data = scipy.io.wavfile.read(os.path.join(path, f))

        abs_data = np.abs(data)
        diff_data = np.diff(data)

        def calc_part_features(data, n=2, prefix=''):
            f_i = 1
            for i in range(0, len(data), len(data)//n):
                features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + len(data)//n])
                features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + len(data)//n])
                features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + len(data)//n])
                features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + len(data)//n])

        features[f]['len'] = len(data)
        if features[f]['len'] > 0:
            n = 1
            calc_part_features(data, n=n)
            calc_part_features(abs_data, n=n, prefix='abs_')
            calc_part_features(diff_data, n=n, prefix='diff_')

            n = 2
            calc_part_features(data, n=n)
            calc_part_features(abs_data, n=n, prefix='abs_')
            calc_part_features(diff_data, n=n, prefix='diff_')

            n = 3
            calc_part_features(data, n=n)
            calc_part_features(abs_data, n=n, prefix='abs_')
            calc_part_features(diff_data, n=n, prefix='diff_')


        cnt += 1

        # if cnt >= 1000:
        #     break

    features = pd.DataFrame(features).T.reset_index()
    features.rename(columns={'index': 'fname'}, inplace=True)
    features.to_csv(file_name, index=False)
    
    return features

def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids
    
if __name__ == "__main__":
    data_path = '../data/'
    ss = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

    audio_train_files = os.listdir('../data/audio_train')
    audio_test_files = os.listdir('../data/audio_test')

    train = pd.read_csv('../data/train.csv')
    submission = pd.read_csv('../data/sample_submission.csv')
    
    train_data = pd.DataFrame()
    train_data['fname'] = train['fname']
    test_data = pd.DataFrame()
    test_data['fname'] = audio_test_files

    train_data = train_data['fname'].progress_apply(get_mfcc, path='../data/audio_train/')
    print('done loading train mfcc')
    test_data = test_data['fname'].progress_apply(get_mfcc, path='../data/audio_test/')
    print('done loading test mfcc')

    train_data['fname'] = train['fname']
    test_data['fname'] = audio_test_files
    train_data['label'] = train['label']
    test_data['label'] = np.zeros((len(audio_test_files)))
    
    path = os.path.join(data_path, 'audio_train')
    train_files = train.fname.values
    train_features = extract_features(train_files, path, 'train')

    path = os.path.join(data_path, 'audio_test')
    test_files = ss.fname.values
    test_features = extract_features(test_files, path, 'test')
    
    train_data = train_data.merge(train_features, on='fname', how='left')
    test_data = test_data.merge(test_features, on='fname', how='left')
    train_data.head()
    
    X = train_data.drop(['label', 'fname'], axis=1)
    feature_names = list(X.columns)
    X = X.values
    labels = np.sort(np.unique(train_data.label.values))
    num_class = len(labels)
    c2i = {}
    i2c = {}
    for i, c in enumerate(labels):
        c2i[c] = i
        i2c[i] = c
    y = np.array([c2i[x] for x in train_data.label.values])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10, shuffle = True)
    clf = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=3000,
                        n_jobs=-1, random_state=0, reg_alpha=0.2, 
                        colsample_bylevel=0.9, colsample_bytree=0.9)
    clf.fit(X_train, y_train)
    val_acc = accuracy_score(clf.predict(X_val), y_val)
    print(val_acc)
    
    #fitting on the entire data
    clf.fit(X, y)
    p = clf.predict_proba(test_data.drop(['label', 'fname'], axis = 1).values)
    np.save('../result/ensembles/xgb_{:.2f}.npy'.format(val_acc), p)