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
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical

from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import kaggle_util
from util import Config, DataGenerator, audio_norm, prepare_data

DEBUG = 0

n_folds = 5
if DEBUG:
    n_folds = 2
    
def get_2d_dummy_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = GlobalMaxPool2D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def get_2d_conv_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def get_2d_conv_model_advance(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def training_2d(train, test):
    config = Config(sampling_rate=44100, audio_duration=2, n_folds=n_folds, 
                    learning_rate=0.001, use_mfcc=True, n_mfcc=40)
    if DEBUG:
        config = Config(sampling_rate=44100, audio_duration=2, n_folds=n_folds, 
                        max_epochs=1, use_mfcc=True, n_mfcc=40)

    X_train = prepare_data(train, config, '../data/audio_train/')
    X_test = prepare_data(test, config, '../data/audio_test/')
    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    
    PREDICTION_FOLDER = "predictions_2d_conv"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
        checkpoint = ModelCheckpoint('../model/best2d_%d.h5'%i, monitor='val_acc', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]
        print("#"*50)
        print("Fold: ", i)
        model = get_2d_conv_model_advance(config)
        history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, 
                            batch_size=64, epochs=config.max_epochs)
        model.load_weights('../model/best2d_%d.h5'%i)
    
        # Save train predictions
        #predictions = model.predict(X_train, batch_size=64, verbose=1)
        #np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)
    
        # Save test predictions
        predictions = model.predict(X_test, batch_size=64, verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)
    
        K.clear_session()
        
    pred_list = []
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
    
    test[['fname', 'label']].to_csv("../result/2d_conv.csv", index=False)
    if not DEBUG:
        command = '/home/kownse/anaconda3/bin/kaggle competitions submit -c freesound-audio-tagging -f ../result/2d_conv.csv -m "submit"'.format(competition, file_7z)
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
        train = train[:2000]
        
    training_2d(train, test)