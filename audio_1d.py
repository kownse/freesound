#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:17:28 2018

@author: kownse
"""

import numpy as np
np.random.seed(1001)

import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.cross_validation import StratifiedKFold

import librosa
import scipy
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution1D, GlobalAveragePooling1D, BatchNormalization, Flatten,
                          GlobalMaxPool1D, MaxPool1D, concatenate, Activation, Concatenate, Input,
                         Dropout, Dense, Add)
from keras.utils import Sequence, to_categorical
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import kaggle_util
from util import Config, DataGenerator, audio_norm, prepare_data

DEBUG = 0

n_folds = 10
if DEBUG:
    n_folds = 2
    
def get_1d_dummy_model(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def get_1d_conv_model(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def conv1d_b(x, filters, kernel_size, padding='valid', strides=1):
    x = Convolution1D(filters, 
                      kernel_size, 
                      padding=padding, 
                      strides = strides,
                      kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x

def conv1d_bn(x, filters, kernel_size, padding='valid', strides=1):
    x = conv1d_b(x, filters, kernel_size, padding, strides)
    x = LeakyReLU(0.1)(x)
    
    return x

def block_inception1d(inp, filters):
    tower_1 = conv1d_bn(inp, filters, 1, padding = 'same')
    tower_1 = conv1d_bn(tower_1, filters, 1, padding = 'same')
    
    tower_2 = conv1d_bn(inp, filters, 1, padding = 'same')
    tower_2 = conv1d_bn(tower_2, filters, 3, padding = 'same')
    tower_2 = conv1d_bn(tower_2, filters, 3, padding = 'same')
    
    tower_3 = conv1d_bn(inp, filters, 1, padding = 'same')
    tower_3 = conv1d_bn(tower_3, filters, 3, padding = 'same')
    tower_3 = conv1d_bn(tower_3, filters, 3, padding = 'same')
    tower_3 = conv1d_bn(tower_3, filters, 3, padding = 'same')
    
    tower_4 = MaxPool1D(3, strides=1, padding='same')(inp)
    tower_4 = conv1d_bn(tower_4, filters, 1, padding = 'same')
    
    x = concatenate([tower_1, tower_2, tower_3, tower_4])
    
    return x

def block_residual(inp, filters, kernel_size, block = 1):
    y = conv1d_bn(inp, filters, kernel_size, padding = 'same')
    y = conv1d_b(y, filters, kernel_size, padding = 'same')
    
    if block == 0:
        shortcut = conv1d_b(inp, filters, kernel_size, padding = 'same')
    else:
        shortcut = inp
        
    y = Add()([y, shortcut])
    y = LeakyReLU(0.1)(y)
    
    return y

def get_1d_conv_inception(config):
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    
    x = block_inception1d(inp, 32)
    x = block_inception1d(x, 32)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.5)(x)
    
    x = block_inception1d(x, 64)
    x = block_inception1d(x, 64)
    
    x = MaxPool1D(8)(x)
    x = Dropout(rate=0.4)(x)
    
    x = block_inception1d(x, 128)
    x = block_inception1d(x, 128)
    
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.3)(x)
    
    x = block_inception1d(x, 256)
    x = block_inception1d(x, 256)
    
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.4)(x)
    
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.1)(x)
    
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def get_1d_conv_head(inp, dropout = True):
    x = conv1d_bn(inp, 32, 9)
    x = conv1d_bn(x, 32, 9)
    x = MaxPool1D(16)(x)
    
    #if dropout:
    #    x = Dropout(rate=0.2)(x)
    
    x = conv1d_bn(x, 64, 6)
    x = conv1d_bn(x, 64, 6)
    x = MaxPool1D(8)(x)
    #if dropout:
    #    x = Dropout(rate=0.2)(x)
    
    x = conv1d_bn(x, 128, 6)
    x = conv1d_bn(x, 128, 6)
    x = MaxPool1D(4)(x)
    #if dropout:
    #    x = Dropout(rate=0.2)(x)
    
    x = conv1d_bn(x, 256, 6)
    x = conv1d_bn(x, 256, 6)
    x = MaxPool1D(4)(x)
    
    #x = conv1d_bn(x, 512, 6)
    #x = conv1d_bn(x, 512, 6)
    #x = MaxPool1D(4)(x)
    
    x = GlobalMaxPool1D()(x)
    #if dropout:
    #    x = Dropout(rate=0.1)(x)
    
    return x

def get_1d_conv_model_advance(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = get_1d_conv_head(inp, False)

    x = Flatten()(x)
    
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.3)(x)
    
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.2)(x)
    
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def get_1d_res_model(config):

    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = conv1d_bn(inp, 16, 3, padding='same')
    x = conv1d_bn(x, 16, 3, padding='same')
    x = conv1d_bn(x, 16, 3, padding='same')
    
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.5)(x)
    
    for i in range(4):
        x = block_residual(x, 32, 3, i)
        #x = Dropout(rate=0.1)(x)
    
    x = MaxPool1D(8)(x)
    x = Dropout(rate=0.4)(x)
    
    for i in range(8):
        x = block_residual(x, 64, 3, i)
        #x = Dropout(rate=0.1)(x)
    
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.3)(x)
    
    for i in range(16):
        x = block_residual(x, 128, 3, i)
        #x = Dropout(rate=0.1)(x)
    
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.5)(x)
    
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.2)(x)
    
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def training_1d(train, test):
    config = Config(sampling_rate=16000, audio_duration=10, n_folds=n_folds, learning_rate=0.001)
    if DEBUG:
        config = Config(sampling_rate=100, audio_duration=1, n_folds=n_folds, max_epochs=1)
        
    
    PREDICTION_FOLDER = "predictions_1d_conv"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)

    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
    
    for i, (train_split, val_split) in enumerate(skf):
        train_set = train.iloc[train_split]
        val_set = train.iloc[val_split]
        checkpoint = ModelCheckpoint('../model/best1d_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        #tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d'%i, write_graph=True)
    
        callbacks_list = [checkpoint, early]#, tb]
        print("Fold: ", i)
        print("#"*50)
        if not DEBUG:
            model = get_1d_conv_model_advance(config)
        else:
            model = get_1d_dummy_model(config)
    
        train_generator = DataGenerator(config, '../data/audio_train/', train_set.index, 
                                        train_set.label_idx, batch_size=64,
                                        preprocessing_fn=audio_norm)
        val_generator = DataGenerator(config, '../data/audio_train/', val_set.index, 
                                      val_set.label_idx, batch_size=64,
                                      preprocessing_fn=audio_norm)
    
        history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                      epochs=config.max_epochs, use_multiprocessing=True, workers=6, max_queue_size=20)
    
        model.load_weights('../model/best1d_%d.h5'%i)
    
        # Save train predictions
        train_generator = DataGenerator(config, '../data/audio_train/', train.index, batch_size=128,
                                        preprocessing_fn=audio_norm)
        predictions = model.predict_generator(train_generator, use_multiprocessing=True, 
                                              workers=6, max_queue_size=20, verbose=1)
        np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)
    
        # Save test predictions
        test_generator = DataGenerator(config, '../data/audio_test/', test.index, batch_size=128,
                                        preprocessing_fn=audio_norm)
        predictions = model.predict_generator(test_generator, use_multiprocessing=True, 
                                              workers=6, max_queue_size=20, verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)
        K.clear_session()
    
    pred_list = []
    for i in range(n_folds):
        pred_list.append(np.load("predictions_1d_conv/test_predictions_%d.npy"%i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('../data/sample_submission.csv')
    test['label'] = predicted_labels
    #print(test)
    print(test[['fname', 'label']].info())
    test[['fname', 'label']].to_csv("../result/1d_conv.csv", index=False)
    if not DEBUG:
        command = '/home/kownse/anaconda3/bin/kaggle competitions submit -c freesound-audio-tagging -f ../result/1d_conv.csv -m "submit"'.format(competition, file_7z)
        os.system(command)


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/sample_submission.csv")
    
    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    if DEBUG:
        train = train[:100]
        #test = test[:2000]
        
    training_1d(train, test)
    
    
    """
    pred_list = []
    for i in range(n_folds):
        pred_list.append(np.load("predictions_1d_conv/test_predictions_%d.npy"%i))
    for i in range(n_folds):
        pred_list.append(np.load("predictions_2d_conv/test_predictions_%d.npy"%i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('../data/sample_submission.csv')
    test['label'] = predicted_labels
    
    kaggle_util.save_result(test[['fname', 'label']],
                           "../result/blend.csv",
                           'freesound-audio-tagging',
                           send = not DEBUG, index=False)
    #test[['fname', 'label']].to_csv("../result/1d_2d_ensembled_submission.csv", index=False)
    """
    