import numpy as np
np.random.seed(1001)

import os
import shutil
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook, tqdm
from sklearn.cross_validation import StratifiedKFold
import librosa
import numpy as np
import scipy
import keras
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
import keras.layers as L
from keras.utils import Sequence, to_categorical
import gc

from util import Config, DataGenerator, audio_norm, getCacheTrainData, getCacheTestData, ensemble
import kaggle_util
from keras import backend as K
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from audio_1d import get_1d_res_model, get_1d_conv_inception, get_1d_conv_model_advance
from dataset import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from util import *
from dataset import *
from cnn2d import *

DEBUG = 0
nfold = 5
if DEBUG:
    nfold = 2
    
train_root = '../data/audio_train/'
test_root = '../data/audio_test/'

def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s
    
def Conv1DTranspose(model, filters, kernel_size, strides=2, padding='same'):
    model.add(L.Lambda(lambda x: K.expand_dims(x, axis=2)))
    model.add(L.Conv2DTranspose(filters=filters, 
                              kernel_size=(kernel_size, 1), 
                              strides=(strides, 1), 
                              padding=padding))
    model.add(L.BatchNormalization())
    model.add(L.LeakyReLU(0.1))
    model.add(L.Lambda(lambda x: K.squeeze(x, axis=2)))
    
    return model

LENGTH_1D = 79872
def get_1d_encoder_model(code_size):
    
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer((LENGTH_1D, 1)))
    
    encoder.add(L.Convolution1D(16, 9, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    #print(encoder.output_shape[1:])
    encoder.add(L.Convolution1D(16, 9, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    encoder.add(L.MaxPool1D(16))
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Convolution1D(32, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    #print(encoder.output_shape[1:])
    encoder.add(L.Convolution1D(32, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    encoder.add(L.MaxPool1D(4))
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Convolution1D(64, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    #print(encoder.output_shape[1:])
    encoder.add(L.Convolution1D(64, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    encoder.add(L.MaxPool1D(4))
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Convolution1D(256, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    #print(encoder.output_shape[1:])
    encoder.add(L.Convolution1D(256, 3, padding="same"))
    encoder.add(L.BatchNormalization())
    encoder.add(LeakyReLU(0.1))
    encoder.add(L.MaxPool1D(4))
    #print(encoder.output_shape[1:])
    encoder_shape = encoder.output_shape[1:]
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))
    
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    
    decoder.add(L.Dense(np.prod(encoder_shape)))
    decoder.add(L.Reshape(encoder_shape))
    #print(decoder.output_shape[1:])
    
    Conv1DTranspose(decoder, 256, 3, strides = 4)
    #print(decoder.output_shape[1:])
    Conv1DTranspose(decoder, 256, 3, strides = 1)
    #print(decoder.output_shape[1:])
    
    Conv1DTranspose(decoder, 64, 3, strides = 4)
    #print(decoder.output_shape[1:])
    Conv1DTranspose(decoder, 256, 3, strides = 1)
    #print(decoder.output_shape[1:])
    
    Conv1DTranspose(decoder, 32, 3, strides = 4)
    #print(decoder.output_shape[1:])
    Conv1DTranspose(decoder, 32, 3, strides = 1)
    #print(decoder.output_shape[1:])
    
    Conv1DTranspose(decoder, 16, 9, strides = 16)
    #print(decoder.output_shape[1:])
    Conv1DTranspose(decoder, 1, 9, strides = 1)
    #print(decoder.output_shape[1:])

    #encoder.summary()
    #decoder.summary()
    return encoder, decoder

MEL_IMGSHAPE = (128, 144, 1)

def get_mel_encoder(code_size):
    img_shape = MEL_IMGSHAPE
    
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(MEL_IMGSHAPE))
    
    ### YOUR CODE HERE: define encoder as per instructions above ###
    encoder.add(L.Conv2D(64, (3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Conv2D(128, (3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Conv2D(256, (3,3), padding='same', activation='elu'))
    encoder.add(L.Conv2D(256, (3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    #print(encoder.output_shape[1:])
    
    encoder.add(L.Conv2D(512, (3,3), padding='same', activation='elu'))
    encoder.add(L.Conv2D(512, (3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    #print(encoder.output_shape[1:])
    encoder_shape = encoder.output_shape[1:]
    
    encoder.add(L.Flatten())                  #flatten image to vector
    encoder.add(L.Dense(code_size))           #actual encoder
    #print(encoder.output_shape[1:])

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    
    decoder.add(L.Dense(np.prod(encoder_shape)))  #actual decoder, height*width*3 units
    decoder.add(L.Reshape(encoder_shape))
    
    decoder.add(L.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=1, activation='elu', padding='same'))
    #print(img_shape, decoder.output_shape[1:])
    #decoder.add(L.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    #print(img_shape, decoder.output_shape[1:])
    
    decoder.add(L.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    #print(img_shape, decoder.output_shape[1:])
    
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    #print(img_shape, decoder.output_shape[1:])
    
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    print(img_shape, decoder.output_shape[1:])
    
    #encoder.summary()
    #decoder.summary()
    return encoder, decoder

def get_autoencoder(code_size, func, input_shape):
    encoder, decoder = func(code_size)
    inp = L.Input(input_shape)
    code = encoder(inp)
    reconstruction = decoder(code)
    
    autoencoder = keras.models.Model(inputs = inp, outputs=reconstruction)
    autoencoder.compile(optimizer="adamax", loss='mse', metrics=['acc'])
    
    #autoencoder.summary()
    return autoencoder, encoder, decoder
    
def fold_train_encoder(X_train, X_test, code_size, model_func, input_shape, flex):
    prefix = "{}folds_{}_{}".format(nfold, config.sampling_rate, config.audio_duration)
    if DEBUG:
        prefix += '_debug'
    PREDICTION_FOLDER = "../result/predictions/encoder_{}_{}".format(flex, prefix)
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
        
    train_pre = None
    test_pre = None

    cvscores = []
    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
    for i, (train_split, val_split) in enumerate(skf):
        checkpath = '../model/encoder_1d_{}.h5' if not DEBUG else '../model/debug1d_{}.h5'
        checkpath = checkpath.format(i)

        checkpoint = ModelCheckpoint(checkpath, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", patience=5)
        callbacks_list = [checkpoint, early]

        print("Fold: ", i)
        print("#"*50)
        model, encoder, decoder = get_autoencoder(code_size, model_func, input_shape)
        print('model getted...')

        x_train_sub = X_train[train_split]  
        x_valid = X_train[val_split]

        model.fit(x_train_sub, x_train_sub, batch_size=64, epochs=config.max_epochs, 
                            validation_data=(x_valid, x_valid), verbose=1,
                            callbacks=callbacks_list)

        
        print(checkpath)
        model.load_weights(checkpath)
        encoder.load_weights(checkpath, by_name = True)
        scores = model.evaluate(x_valid, x_valid)
        print(scores[1])
        cvscores.append(scores[1])
        
        # Save train predictions
        print('save result npy')
        predictions = encoder.predict(X_test, batch_size = 64, verbose = 1) 
        if test_pre is None:
            test_pre = predictions / nfold
        else:
            test_pre += predictions / nfold
        np.save(PREDICTION_FOLDER + '/code_test_sub_{}_{}'.format(flex, i), predictions)

        predictions = encoder.predict(X_train, batch_size = 64, verbose = 1) 
        if train_pre is None:
            train_pre = predictions / nfold
        else:
            train_pre += predictions / nfold
        np.save(PREDICTION_FOLDER + '/code_train_sub_{}_{}'.format(flex, i), predictions)
        K.clear_session()

    cvmean = np.mean(cvscores)
    cvstd = np.std(cvscores)
    np.save(PREDICTION_FOLDER + '/code_train_{}_{}'.format(flex, code_size), train_pre)
    np.save(PREDICTION_FOLDER + '/code_test_{}_{}'.format(flex, code_size), test_pre)
    
def minmax(data):
    data_min = data.min()
    data_max = data.max()
    gap = data_max - data_min
    for i in tqdm(range(data.shape[0])):
        data[i] = (data[i] - data_min) / gap
        
    return data
    
def encoder_1d(train, test, code_size):

    X_train, y = getCacheTrainData(train, train_root, 
                                   config.sampling_rate, config.audio_duration, flex='')
    X_test = getCacheTestData(test, test_root, config.sampling_rate, config.audio_duration,flex='')

    X_train = X_train[:,:LENGTH_1D, :]
    X_test = X_test[:,:LENGTH_1D, :]
    
    if DEBUG:
        X_train = X_train[:1000]
        X_test = X_test[:1000]
        
    X_train = minmax(X_train)
    X_test = minmax(X_test)
    
    fold_train_encoder(X_train, X_test, code_size, get_1d_encoder_model, (LENGTH_1D,1), '1d')
    
def reshape_mel(data):
    from skimage.transform import resize
    
    shape = data.shape
    #tmp_shape = (shape[0], shape[1], 144, shape[3])
    tmp_shape = (shape[0], MEL_IMGSHAPE[0], MEL_IMGSHAPE[1], MEL_IMGSHAPE[2])
    out = np.zeros(tmp_shape)
    
    for i in tqdm(range(tmp_shape[0])):
        out[i] = resize(data[i], tmp_shape[1:], 
                        preserve_range=True, 
                        mode='reflect')
        
    return minmax(out)
    
    
def encoder_mel(train, test, code_size):
    X_train_mel, y = getMelTrainData(train, train_root, config)
    X_test_mel = getMelTestData(test, test_root, config)
    if DEBUG:
        X_train_mel = X_train_mel[:1000]
        X_test_mel = X_test_mel[:1000]
        
    X_train = reshape_mel(X_train_mel)
    X_test = reshape_mel(X_test_mel)
    
    del X_train_mel, X_test_mel; gc.collect()
    fold_train_encoder(X_train, X_test, code_size, get_mel_encoder, MEL_IMGSHAPE, 'mel')
 
def encoder_mfcc(train, test, code_size):
    X_train_mfcc = getCacheMFCC(train, config, train_root,'train')
    X_test_mfcc = getCacheMFCC(test, config, test_root,'test')
    if DEBUG:
        X_train_mfcc = X_train_mfcc[:1000]
        X_test_mfcc = X_test_mfcc[:1000]

    X_train = reshape_mel(X_train_mfcc)
    X_test = reshape_mel(X_test_mfcc)
    
    del X_train_mfcc, X_test_mfcc; gc.collect()
    fold_train_encoder(X_train, X_test, code_size, get_mel_encoder, MEL_IMGSHAPE, 'mfcc')

    
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
            audio_duration=5, n_folds=nfold, learning_rate=0.0001,
            use_mfcc = True
        )
    if DEBUG:
        config.max_epochs = 5
    
    encoder_mfcc(train, test, 64)
    encoder_mfcc(train, test, 128)
    encoder_1d(train, test, 64)
    encoder_1d(train, test, 128)
    #encoder_mel(train, test, 64)
    