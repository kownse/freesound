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

DEBUG = 1

n_folds = 5
if DEBUG:
    n_folds = 2
    
def get_rnn_model(config):
    
    from keras.layers import \
    BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, LSTM, GRU, TimeDistributed, Bidirectional, GlobalMaxPooling1D, Conv1D
    
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    #x = Bidirectional(LSTM(256, return_sequences=True))(inp)
    avg_pool_rnn = GlobalAveragePooling1D()(x)
    max_pool_rnn = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool_rnn, max_pool_rnn])
    
    x = Dropout(0.5)(x)
    
    #out = TimeDistributed(Dense(nclass, activation=softmax))(x)
    #out = Activation('softmax')(x)
    #x = Dense(256, activation=relu)(x)
    #x = Dropout(rate=0.3)(x)
    #x = Dense(1028, activation=relu)(x)
    #x = Dropout(rate=0.5)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def training_rnn(train, test):
    config = Config(sampling_rate=800, audio_duration=10, n_folds=n_folds, learning_rate=0.001)
    if DEBUG:
        config = Config(sampling_rate=100, audio_duration=1, n_folds=n_folds, max_epochs=1)
        
    PREDICTION_FOLDER = "predictions_rnn"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
    
    for i, (train_split, val_split) in enumerate(skf):
        train_set = train.iloc[train_split]
        val_set = train.iloc[val_split]
        checkpoint = ModelCheckpoint('../model/bestrnn_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d'%i, write_graph=True)
    
        callbacks_list = [checkpoint, early, tb]
        print("Fold: ", i)
        print("#"*50)
        model = get_rnn_model(config)
    
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
    
        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)
    
    
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
    test[['fname', 'label']].to_csv("../result/rnn_ensembled_submission.csv", index=False)

    
    
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
        
    training_rnn(train, test)