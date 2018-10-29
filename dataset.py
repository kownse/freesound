from keras.utils import Sequence, to_categorical
from util import Config
from transform import *
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
pool_size = 6  # your "parallelness"

def recude_mem_usage_np(data):
    c_min = data.min()
    c_max = data.max()
    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        return data.astype(np.float16)
    else:
        return data

def process_data(data, callbacks):
    for func in callbacks:
        data = func(data)
    return data

def prepare_wav(path, callbacks):
    data = {'path':path}
    return process_data(data, callbacks)

def mixup_onedata(data, labels, weight, index, batch_size):
    x = np.zeros_like(data, dtype=data.dtype)
    y = np.zeros_like(labels, dtype=labels.dtype)
    
    x1, x2 = data, data[index]
    y1, y2 = labels, labels[index]
    
    for i in tqdm(range(batch_size)):
        x[i] = x1[i] * weight[i] + x2[i] * (1 - weight[i])
        y[i] = y1[i] * weight[i] + y2[i] * (1 - weight[i])
    return x, y

def mixup_all(data, labels, alpha):
    batch_size = len(labels)
    weight = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    
    return mixup_onedata(data, labels, weight, index, batch_size)

def mixup_bulk(datas, labels, alpha):
    batch_size = len(labels)
    weight = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    
    y = np.zeros_like(labels, dtype=np.float16)
    y1, y2 = labels, labels[index]
    for i in tqdm(range(batch_size)):
        y[i] = y1[i] * weight[i] + y2[i] * (1 - weight[i])
    
    xx = []
    for data in datas:
        x1, x2 = data, data[index]
        x = np.zeros_like(data, dtype=np.float16)
        for i in tqdm(range(batch_size)):
            x[i] = x1[i] * weight[i] + x2[i] * (1 - weight[i])
        xx.append(x)
    return xx, y

class MelDataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, argument = False, padfunc = 'constant'):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()
        self.dim = self.config.dim
        
        data_aug_transform = []
        if argument is True:
            data_aug_transform += [
                ChangeAmplitude(), 
                ChangeSpeedAndPitchAudio(), 
                StretchAudio(),
                TimeshiftAudio(),
            ]
        data_aug_transform += [FixAudioLength(config.audio_duration, padfunc)]
        self.callbacks = [LoadAudio(config.sampling_rate)] + data_aug_transform
        self.analyze = [
            ToMelSpectrogram(n_mels=config.n_mels),
            ToMFCC(n_mfcc = config.n_mfcc)
        ]
        
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)
    
    def getitem_all(self):
        return self.__data_generation(self.list_IDs, True)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        
    def normalize(self, X):
        if self.config.postfunc == 'normalize':
            X = (X - X.mean()) / X.std()
        elif self.config.postfunc == 'minmax':
            X = (X - X.mean()) / (X.max() - X.min())
        return X    

    def __data_generation(self, list_IDs_temp, progress=False):
        cur_batch_size = len(list_IDs_temp)

        datas = []
        input_length = self.config.audio_length
        if progress:
            pbar = tqdm(total=cur_batch_size)
        
        
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            data =  prepare_wav(file_path, self.callbacks)
            #data['samples_original'] = data['samples'].copy()
            datas.append(data)

            if progress:
                pbar.update(1)

        X_mel = None
        X_mfcc = None
        X_wav = None

        if progress:
            pbar.close()

        if progress:
            pbar = tqdm(total=cur_batch_size)
        for i in range(len(datas)):
            data = process_data(datas[i], self.analyze)
            wav = data['samples']
            if X_wav is None:
                X_wav = np.empty((cur_batch_size, wav.shape[0]), dtype=np.float32)
            X_wav[i] = wav
            
            mel = data['mel_spectrogram']
            if X_mel is None:
                X_mel = np.empty((cur_batch_size, mel.shape[0], mel.shape[1]), dtype=np.float32)
            X_mel[i] = mel
            
            mfcc = data['mfcc']
            if X_mfcc is None:
                X_mfcc = np.empty((cur_batch_size, mfcc.shape[0], mfcc.shape[1]), dtype=np.float32)
            X_mfcc[i] = mfcc

            if progress:
                pbar.update(1)

        if progress:
            pbar.close()

        X_mel = np.expand_dims(X_mel, -1)
        X_mfcc = np.expand_dims(X_mfcc, -1)
        X_wav = np.expand_dims(X_wav, -1)
        
        X_mel = recude_mem_usage_np(X_mel)
        X_mfcc = recude_mem_usage_np(X_mfcc)
        X_wav = recude_mem_usage_np(X_wav)
        
        y = None
        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            y = to_categorical(y, num_classes=self.config.n_classes).astype(np.uint8)

        if self.labels is not None:
            return X_wav, X_mel, X_mfcc, y
        else:
            return X_wav, X_mel, X_mfcc
        
class WavDataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, argument = False, padfunc = 'constant'):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()
        self.dim = self.config.dim
        
        data_aug_transform = []
        if argument is True:
            data_aug_transform += [
                ChangeAmplitude(), 
                ChangeSpeedAndPitchAudio(), 
                StretchAudio(),
                TimeshiftAudio(),
            ]
        data_aug_transform += [FixAudioLength(config.audio_duration, padfunc)]
        self.callbacks = [LoadAudio(config.sampling_rate)] + data_aug_transform
        
    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)
    
    def getitem_all(self):
        return self.__data_generation(self.list_IDs, True)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        
    def normalize(self, X):
        if self.config.postfunc == 'normalize':
            X = (X - X.mean()) / X.std()
        elif self.config.postfunc == 'minmax':
            X = (X - X.mean()) / (X.max() - X.min())
        return X    

    def __data_generation(self, list_IDs_temp, progress=False):
        cur_batch_size = len(list_IDs_temp)

        datas = []
        input_length = self.config.audio_length
        if progress:
            pbar = tqdm(total=cur_batch_size)
        
        
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            data =  prepare_wav(file_path, self.callbacks)
            #data['samples_original'] = data['samples'].copy()
            datas.append(data)

            if progress:
                pbar.update(1)

        X_wav = None

        if progress:
            pbar.close()

        if progress:
            pbar = tqdm(total=cur_batch_size)
        for i in range(len(datas)):
            data = datas[i]
            wav = data['samples']
            if X_wav is None:
                X_wav = np.empty((cur_batch_size, wav.shape[0]), dtype=np.float32)
            X_wav[i] = wav

            if progress:
                pbar.update(1)

        if progress:
            pbar.close()

        X_wav = np.expand_dims(X_wav, -1)
        X_wav = recude_mem_usage_np(X_wav)
        
        y = None
        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            y = to_categorical(y, num_classes=self.config.n_classes).astype(np.uint8)

        if self.labels is not None:
            return X_wav, y
        else:
            return X_wav
        
def getMelTrainData(train, path, config):
    flex = '{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(train), config.n_mels, config.postfunc)
    
    train_x_path = '../cache/mel_train_x_{}.npy'.format(flex)
    train_y_path = '../cache/mel_train_y_{}.npy'.format(flex)
    
    
    if os.path.exists(train_x_path) and os.path.exists(train_y_path):
        X_train = np.load(train_x_path)
        y = np.load(train_y_path)
    else:
        print(train_x_path, 'not exist')
        train_generator = MelDataGenerator(config, path, train.index, 
                                            train.label_idx, batch_size=64)
        X_train,y = train_generator.getitem_all()
        np.save(train_x_path, X_train)
        np.save(train_y_path, y)
        
    return X_train, y

def getMelTestData(test, path, config):
    flex = '{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(test), config.n_mels, config.postfunc)
    
    test_x_path = '../cache/mel_test_{}.npy'.format(flex)

    if os.path.exists(test_x_path):
        X_test = np.load(test_x_path)
    else:
        print(test_x_path, 'not exist')
        train_generator = MelDataGenerator(config, path, test.index, 
                                            batch_size=64)
        X_test = train_generator.getitem_all()
        np.save(test_x_path, X_test)
        
    return X_test

def get_cachedata_all_train(train, train_root, config, argu_cnt, padfunc):
    padflex = '' if padfunc == 'constant' else padfunc
    flex = '{}_{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(train), config.n_mels, argu_cnt, padflex)
    if 'trimmed' in train_root:
        flex += '_trim'
    
    wav_path = '../cache/wav_train_{}.npy'.format(flex)
    mel_path = '../cache/mel_train_{}.npy'.format(flex)
    mfcc_path = '../cache/mfcc_train_{}.npy'.format(flex)
    train_y_path = '../cache/mel_train_y_{}.npy'.format(flex)
    
    print(wav_path)
    
    if os.path.exists(wav_path) and\
        os.path.exists(train_y_path) and\
        os.path.exists(mel_path) and\
        os.path.exists(mfcc_path):
        X_wav = np.load(wav_path)
        X_mel = np.load(mel_path)
        X_mfcc = np.load(mfcc_path)
        y = np.load(train_y_path)
    else:
        generator = MelDataGenerator(config, train_root, train.index, 
                                     train.label_idx, batch_size=64, padfunc=padfunc)
        X_wav, X_mel, X_mfcc, y= generator.getitem_all()
        
        argu_generator = MelDataGenerator(config, train_root, train.index, 
                                          train.label_idx, batch_size=64,
                                          argument = True)
        for i in range(argu_cnt):
            X_wav_argu, X_mel_argu, X_mfcc_argu,y_argu = argu_generator.getitem_all()
            X_wav = np.vstack([X_wav, X_wav_argu])
            X_mel = np.vstack([X_mel, X_mel_argu])
            X_mfcc = np.vstack([X_mfcc, X_mfcc_argu])
            y = np.vstack([y, y_argu])
    
        np.save(wav_path, X_wav)
        np.save(mel_path, X_mel)
        np.save(mfcc_path, X_mfcc)
        np.save(train_y_path, y)
        
    X_wav = recude_mem_usage_np(X_wav)
    X_mel = recude_mem_usage_np(X_mel)
    X_mfcc = recude_mem_usage_np(X_mfcc)
    y = recude_mem_usage_np(y)
    print('\nwav dtype {}\nmel dtype {}\nmfcc dtype {}'.format(X_wav.dtype,
                                                              X_mel.dtype,
                                                              X_mfcc.dtype))
    return X_wav, X_mel, X_mfcc, y

def get_cachedata_all_test(test, test_root, config, padfunc):
    padflex = '' if padfunc == 'constant' else padfunc
    flex = '{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(test), config.n_mels, padflex)
    if 'trimmed' in test_root:
        flex += '_trim'
    
    wav_path = '../cache/wav_test_{}.npy'.format(flex)
    mel_path = '../cache/mel_test_{}.npy'.format(flex)
    mfcc_path = '../cache/mfcc_test_{}.npy'.format(flex)
    
    if os.path.exists(wav_path) and\
        os.path.exists(mel_path) and\
        os.path.exists(mfcc_path):
        
        X_wav = np.load(wav_path)
        X_mel = np.load(mel_path)
        X_mfcc = np.load(mfcc_path)
    else:
        generator = MelDataGenerator(config, test_root, test.index, batch_size=64, padfunc=padfunc)
        X_wav, X_mel, X_mfcc = generator.getitem_all()
        
        np.save(wav_path, X_wav)
        np.save(mel_path, X_mel)
        np.save(mfcc_path, X_mfcc)
        
    X_wav = recude_mem_usage_np(X_wav)
    X_mel = recude_mem_usage_np(X_mel)
    X_mfcc = recude_mem_usage_np(X_mfcc)
    print('\nwav dtype {}\nmel dtype {}\nmfcc dtype {}'.format(X_wav.dtype,
                                                              X_mel.dtype,
                                                              X_mfcc.dtype))
    return X_wav, X_mel, X_mfcc

def get_cachedata_wav_train(train, train_root, config, argu_cnt, padfunc):
    padflex = '' if padfunc == 'constant' else padfunc
    flex = '{}_{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(train), config.n_mels, argu_cnt, padflex)
    if 'trimmed' in train_root:
        flex += '_trim'
    
    wav_path = '../cache/wav_train_{}.npy'.format(flex)
    train_y_path = '../cache/mel_train_y_{}.npy'.format(flex)
    
    print(wav_path)
    
    if os.path.exists(wav_path) and\
        os.path.exists(train_y_path):
        X_wav = np.load(wav_path)
        y = np.load(train_y_path)
    else:
        generator = WavDataGenerator(config, train_root, train.index, 
                                     train.label_idx, batch_size=64, padfunc=padfunc)
        X_wav, y= generator.getitem_all()
        
        argu_generator = WavDataGenerator(config, train_root, train.index, 
                                          train.label_idx, batch_size=64,
                                          argument = True)
        for i in range(argu_cnt):
            X_wav_argu, y_argu = argu_generator.getitem_all()
            X_wav = np.vstack([X_wav, X_wav_argu])
            y = np.vstack([y, y_argu])
    
        np.save(wav_path, X_wav)
        np.save(train_y_path, y)
        
    return X_wav, y

def get_cachedata_wav_test(test, test_root, config, padfunc):
    padflex = '' if padfunc == 'constant' else padfunc
    flex = '{}_{}_{}_{}{}'.format(config.sampling_rate, config.audio_duration,
                                len(test), config.n_mels, padflex)
    if 'trimmed' in test_root:
        flex += '_trim'
    
    wav_path = '../cache/wav_test_{}.npy'.format(flex)
    
    if os.path.exists(wav_path):
        
        X_wav = np.load(wav_path)
    else:
        generator = WavDataGenerator(config, test_root, test.index, batch_size=64, padfunc=padfunc)
        X_wav = generator.getitem_all()
        
        np.save(wav_path, X_wav)
        
    return X_wav

def combine_normalize(data1, data2):
    len_data1 = len(data1)
    data_all = np.vstack([data1, data2]).astype(np.float32)
    print(data_all.mean(), data_all.std())
    data_all = (data_all - data_all.mean()) / data_all.std()
    print(data_all.mean(), data_all.std())
    data1 = data_all[:len_data1]
    data2 = data_all[len_data1:]
    return data1, data2