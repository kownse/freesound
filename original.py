import numpy as np
np.random.seed(1001)

import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.cross_validation import StratifiedKFold

import librosa
import numpy as np
import scipy

from keras.utils import Sequence, to_categorical
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from util import *
import kaggle_util
from keras import backend as K
import tensorflow as tf
from audio_1d import get_1d_res_model, get_1d_conv_inception, get_1d_conv_model_advance
from dataset import *
import gc

DEBUG = 0
nfold = 10
if DEBUG:
    nfold = 2
    
train_root = '../data/audio_train/'
test_root = '../data/audio_test/'

#train_root = '../data/audio_train_trimmed/'
#test_root = '../data/audio_test_trimmed/'
  
if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv", index_col="fname")
    test = pd.read_csv("../data/sample_submission.csv", index_col="fname")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    if DEBUG:
        train = train[:10]
        test = test[:10]
    
    config = Config(
        sampling_rate=16000, n_mels = 128, n_mfcc = 128,
        audio_duration=5, n_folds=nfold, learning_rate=0.001,
        postfunc='', mixup=2, mixup_alpha = 2,
        use_mfcc = True
    )
    #config.n_folds = 2
    #config.max_epochs = 1
    if DEBUG:
        config.max_epochs = 1
    """
    X_train, y = getCacheTrainData(train, train_root, 
                                   config.sampling_rate, config.audio_duration, flex='trim')
    X_test = getCacheTestData(test, test_root, config.sampling_rate, config.audio_duration,flex='trim')
    argucnt = 0
    """
    
    argucnt = 3
    padfunc = 'constant'
    X_train_original, y =\
            get_cachedata_wav_train(train, train_root, config, argucnt, padfunc=padfunc)
    X_test_original =\
        get_cachedata_wav_test(test, test_root, config, padfunc=padfunc)
        
    X_train = X_train_original
    X_test = X_test_original

    y_label = np.load('../cache/train_y.npy')
    if DEBUG:
        y_label = y_label[:10]
    y_label = np.argmax(y_label, axis=1)
    len_y = len(y_label)
    
    X_train = X_train[:len_y*2]
        
    prefix = "{}folds_{}_{}".format(nfold, config.sampling_rate, config.audio_duration)
    if DEBUG:
        prefix += '_debug'
    PREDICTION_FOLDER = "../result/predictions/cnn1d_{}".format(prefix)
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
    for idx in range(1):
        print('total folds ', config.n_folds)
        cvscores = []
        skf = StratifiedKFold(y_label, n_folds=config.n_folds)
        for i, (train_split, val_split) in enumerate(skf):
            if i < 2:
                continue
            
            if argucnt > 0:
                train_split = np.hstack([(train_split + len_y * i) for i in range(argucnt)])
            checkpath = '../model/best1d_{}.h5' if not DEBUG else '../model/debug1d_{}.h5'
            checkpath = checkpath.format(i)

            checkpoint = ModelCheckpoint(checkpath, monitor='val_acc', verbose=1, save_best_only=True)
            early = EarlyStopping(monitor="val_acc", patience=5)
            callbacks_list = [checkpoint, early]

            print("Fold: ", i)
            print("#"*50)
            model = get_1d_conv_model_advance(config)
            print('model getted...')

            x_train_sub = X_train[train_split]
            y_train_sub = y[train_split]
            
            if config.mixup > 0:
                cnn1d_sub = None
                y_tmp_sub = None
                originals = [x_train_sub]
                for j in range(config.mixup):
                    print('mixup', j)
                    [cnn1d_tmp], y_tmp = mixup_bulk(originals, y_train_sub, config.mixup_alpha)

                    cnn1d_sub = cnn1d_tmp if cnn1d_sub is None else np.vstack((cnn1d_sub, cnn1d_tmp))
                    del cnn1d_tmp; gc.collect()
                    y_tmp_sub = y_tmp if y_tmp_sub is None else np.vstack((y_tmp_sub, y_tmp))
                    del y_tmp; gc.collect()
                x_train_sub = cnn1d_sub
                y_train_sub = y_tmp_sub
            print('train length: ', len(x_train_sub), len(y_train_sub))    
            x_valid = X_train[val_split]
            y_valid = y[val_split]
            
            model.fit(x_train_sub, y_train_sub, batch_size=64, epochs=config.max_epochs, 
                                validation_data=(x_valid, y_valid), verbose=1,
                                callbacks=callbacks_list)

            model.load_weights(checkpath)
            
            p = model.predict(x_valid, batch_size = 64, verbose = 1)
            valid_score = get_valid_score(y_valid, p)
            print("Score = {:.4f}".format(valid_score))
            cvscores.append(valid_score)
            
            #scores = model.evaluate(x_valid, y_valid)
            #print(scores[1])
            #cvscores.append(scores[1])
            
            # Save train predictions
            print('save result npy')
            predictions = model.predict(X_test, batch_size = 64, verbose = 1)

            savepath = "/p{}.npy"
            savepath = savepath.format(i)
            np.save(PREDICTION_FOLDER + savepath, predictions)
            K.clear_session()
            #break
        cvmean = np.mean(cvscores)
        cvstd = np.std(cvscores)
        print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
        actual_prefix = prefix + '{0:.3f}_{1:.3f}'.format(cvmean, cvstd)
        ensemble(LABELS, config.n_folds, [PREDICTION_FOLDER], actual_prefix, 'cnn1d', False)
