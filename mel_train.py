import numpy as np
#np.random.seed(1001)

import os
import shutil
import pandas as pd
import gc
from util import *
from dataset import *
from cnn2d import *
from seresnet import *
from analyze import *
import cnn2d

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
import keras

import logging

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet201

LOG_FILENAME = 'train.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

DEBUG = 0
nfold = 10
if DEBUG:
    nfold = 2
    
train_root = '../data/audio_train/'
test_root = '../data/audio_test/'
#train_root = '../data/audio_train_trimmed/'
#test_root = '../data/audio_test_trimmed/'

models = [
    ('vgg11', [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']),
    ('vgg13', [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],),
    ('seresnet18', []),
    ('seresnet34', []),
    ('seresnet50', []),
    ('numeric', []),
    ('2rdcnn', [])
]

def combine_minmax(train, test):
    len_train = len(train)
    all = np.vstack([train, test])
    tmp = (all - all.min()) / (all.max() - all.min()) - 0.5
    return tmp[:len_train], tmp[len_train:]

def make_model(config, mname, cfg, shape, dense_cfg):
    print(mname)
    length = shape[2]
    if 'pre' in mname:
        if 'vgg16' in mname:
            backend = VGG16(weights='imagenet', include_top=False, input_shape = (128, 157, 3))
        elif 'vgg19' in mname:
            backend = VGG19(weights='imagenet', include_top=False, input_shape = (128, 157, 3))
        elif 'xception' in mname:
            backend = Xception(weights='imagenet', include_top=False, input_shape = (128, 157, 3))
        elif 'resnet50' in mname:
            backend = ResNet50(weights='imagenet', include_top=False, input_shape = (197, 197, 3))
        elif 'inception_v3' in mname:
            backend = InceptionV3(weights='imagenet', include_top=False, input_shape = (139, 157, 3))
        elif 'inception_resnet_v2' in mname:
            backend = InceptionResNetV2(weights='imagenet', include_top=False, input_shape = (139, 157, 3))
        elif 'mobilenet' in mname:
            backend = MobileNet(weights='imagenet', include_top=False, input_shape = (160, 160, 3))
        elif 'densenet121' in mname:
            backend = DenseNet121(weights='imagenet', include_top=False, input_shape = (221, 221, 3))
        elif 'densenet201' in mname:
            backend = DenseNet201(weights='imagenet', include_top=False, input_shape = (221, 221, 3))
        
        else:
            print('\npretrained model not found\n')
            
        out = dense_outlayer(backend.output, config.n_classes, dense_cfg, flatten = True)
        return cnn2d.make_model(config, backend.input, out)
       
    elif 'vgg' in mname:
        return make_vgg(config, cfg, length, True, dense_cfg)
    elif 'seresnet' in mname:
        return make_seresnet(mname, config, length, dense_cfg)
    elif 'resnet' in mname:
        return make_resnet(config, cfg, length)
    elif 'resnext' in mname:
        return make_resnext(config, length)
    elif 'numeric' in mname:
        length = shape[1]
        return make_numeric(config, cfg, length, dense_cfg)
    elif '2rdcnn' in mname:
        return make_2rdcnn(config, cfg, length, dense_cfg)

def getSubData(data, cols, index):
    sub_data = {}
    for col in cols:
        sub_data[col] = data[col][index]
    return sub_data

def retrain_sgd(y_label, argucnt, config, X_train, y, X_test, dense_cfg):
    batchsize = 64
    print('batchsize', batchsize)
    cvscores = []    
    print('total folds ', config.n_folds, argucnt)
    len_y = len(y_label)
    skf = StratifiedKFold(y_label, n_folds=config.n_folds, random_state = 5)
    for i, (train_split, val_split) in enumerate(skf):
        train_split = np.hstack([(train_split + len_y * i) for i in range(argucnt)])
        print(len(train_split), len(val_split))

        config.opt = 'sgd'
        model = make_model(config, config.data_mname, config.cfg, X_train.shape, dense_cfg)
        
        checkpath = '../model/best1d_{}{}.h5' if not DEBUG else '../model/debug1d_{}{}.h5'
        checkpath = checkpath.format(config.data_mname, i)
        print('original path: ', checkpath)
        model.load_weights(checkpath)
        del checkpath
        
        checkpath_sgd = '../model/best1d_{}{}_sgd.h5' if not DEBUG else '../model/debug1d_{}{}.h5'
        checkpath_sgd = checkpath_sgd.format(config.data_mname, i)
        
        #model.summary()
        #exit()
        print('model getted...')
        print("Fold: ", i)
        print("#"*50)
        x_valid = X_train[val_split]
        y_valid = y[val_split]
        
        
        scores = model.evaluate(x_valid, y_valid)
        print('before sgd ', scores[1])

        #print(x_train_sub.shape, x_valid.shape)
        if i > -1:
            print('mixup for fold ', i)
            x_original = X_train[train_split]
            y_original = y[train_split]
            if config.mixup > 0:
                x_train_sub = None
                y_train_sub = None
                for j in range(config.mixup):
                    print('mixup', config.mixup)
                    x_tmp, y_tmp = mixup_all(x_original, y_original, config.mixup_alpha)
                    x_train_sub = x_tmp if x_train_sub is None else np.vstack((x_train_sub, x_tmp))
                    y_train_sub = y_tmp if y_train_sub is None else np.vstack((y_train_sub, y_tmp))
            else:
                x_train_sub = x_original
                y_train_sub = y_original

            print(x_train_sub.shape, y_train_sub.shape)
        
            checkpoint = ModelCheckpoint(checkpath_sgd, monitor='val_acc', verbose=1, save_best_only=True)
            early = EarlyStopping(monitor="val_acc", patience=4)
            callbacks_list = [checkpoint, early]
            if not DEBUG:
                tensorboard = TensorBoard(log_dir='./graph/{}_{}_sgd'.format( config.data_mname, 'f'), histogram_freq=0,  write_graph=True, write_images=True)
                callbacks_list.append(tensorboard)
            model.fit(x_train_sub, y_train_sub, batch_size=batchsize, epochs=config.max_epochs, 
                            validation_data=(x_valid, y_valid), verbose=1,
                            callbacks=callbacks_list)
        model.load_weights(checkpath_sgd)

        scores = model.evaluate(x_valid, y_valid)
        logger.info('{}_{} val_loss: {:.4f} val_acc: {:.4f}'.format(
            config.data_mname,config.prefix, scores[0], scores[1]))
        print(scores[1])
        cvscores.append(scores[1])
        # Save train predictions
        print('save result npy')
        
        predictions = model.predict(X_test, batch_size = batchsize, verbose = 1)
        savepath = "/p{}.npy"
        savepath = savepath.format(i)
        np.save(config.PREDICTION_FOLDER + savepath, predictions)

        K.clear_session()
    
    cvmean = np.mean(cvscores)
    cvstd = np.std(cvscores)
    print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
    actual_prefix = config.prefix + '{0:.3f}_{1:.3f}'.format(cvmean, cvstd)
    ensemble(LABELS, config.n_folds, [config.PREDICTION_FOLDER], actual_prefix, config.data_mname, False)
       
    
def oneturn_model(y_label, argucnt, config, X_train, y, X_test, dense_cfg):
    batchsize = 32
    print('batchsize', batchsize)
    cvscores = []    
    print('total folds ', config.n_folds, argucnt)
    len_y = len(y_label)
    skf = StratifiedKFold(y_label, n_folds=config.n_folds, random_state = 5)
    for i, (train_split, val_split) in enumerate(skf):
        train_split = np.hstack([(train_split + len_y * i) for i in range(argucnt + 1)])
        print(len(train_split), len(val_split))

        checkpath = '../model/best1d_{}{}.h5' if not DEBUG else '../model/debug1d_{}{}.h5'
        checkpath = checkpath.format(config.data_mname, i)
        model = make_model(config, config.data_mname, config.cfg, X_train.shape, dense_cfg)

        #model.summary()
        #exit()
        print('model getted...')
        print("Fold: ", i)
        print("#"*50)
        
        x_valid = X_train[val_split]
        y_valid = y[val_split]
        #print(x_train_sub.shape, x_valid.shape)
        if i > 0:
            print('mixup for fold ', i)
            x_original = X_train[train_split]
            y_original = y[train_split]
            if config.mixup > 0:
                x_train_sub = None
                y_train_sub = None
                for j in range(config.mixup):
                    print('mixup', config.mixup)
                    x_tmp, y_tmp = mixup_all(x_original, y_original, config.mixup_alpha)
                    x_train_sub = x_tmp if x_train_sub is None else np.vstack((x_train_sub, x_tmp))
                    y_train_sub = y_tmp if y_train_sub is None else np.vstack((y_train_sub, y_tmp))
            else:
                x_train_sub = x_original
                y_train_sub = y_original

            print(x_train_sub.shape, y_train_sub.shape)
        
            checkpoint = ModelCheckpoint(checkpath, monitor='val_acc', verbose=1, save_best_only=True)
            early = EarlyStopping(monitor="val_acc", patience=5)
            callbacks_list = [checkpoint, early]
            if not DEBUG:
                tensorboard = TensorBoard(log_dir='./graph/{}_{}_noarg'.format( config.data_mname, 'f'), histogram_freq=0,  write_graph=True, write_images=True)
                callbacks_list.append(tensorboard)
            model.fit(x_train_sub, y_train_sub, batch_size=batchsize, epochs=config.max_epochs, 
                            validation_data=(x_valid, y_valid), verbose=1,
                            callbacks=callbacks_list)
        model.load_weights(checkpath)

        scores = model.evaluate(x_valid, y_valid)
        logger.info('{}_{} val_loss: {:.4f} val_acc: {:.4f}'.format(
            config.data_mname,config.prefix, scores[0], scores[1]))
        print(scores[1])
        cvscores.append(scores[1])
        # Save train predictions
        print('save result npy')
        
        predictions = model.predict(X_test, batch_size = batchsize, verbose = 1)
        savepath = "/p{}.npy"
        savepath = savepath.format(i)
        np.save(config.PREDICTION_FOLDER + savepath, predictions)

        K.clear_session()
    
    cvmean = np.mean(cvscores)
    cvstd = np.std(cvscores)
    print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
    actual_prefix = config.prefix + '{0:.3f}_{1:.3f}'.format(cvmean, cvstd)
    ensemble(LABELS, config.n_folds, [config.PREDICTION_FOLDER], actual_prefix, config.data_mname, False)
        
def single_models(LABELS, train, test):
    cfgs = [
        ('mel', 5, 128, '', 2, 2, 3, ('pre_xception', []), 1, [(256, 0.5),(64, 0.2)], 'pre_xception_mix2', 'constant', 'sgd'),
        ('mfcc', 5, 128, '', 2, 2, 3, ('pre_xception', []), 1, [(256, 0.5),(64, 0.2)], 'pre_xception_mix32', 'constant', 'train'),
        ('mfcc', 5, 128, '', 2, 2, 3, ('pre_xception', []), 1, [(256, 0.5),(64, 0.2)], 'pre_xception_mix32', 'constant', 'sgd'),
        #('mel', 5, 128, '', 0, 2, 3, ('pre_densenet121', []), 1, [(256, 0.5),(64, 0.2)], 'densenet121', 'constant'),
        #('mel', 5, 128, '', 0, 2, 3, ('pre_densenet201', []), 1, [(256, 0.5),(64, 0.2)], 'densenet201', 'constant'),
        #('mel', 5, 128, '', 0, 2, 3, models[0], 1, [(256, 0.5), (64, 0.2)], 'no_mix_trim', 'constant'),
        #('mfcc', 5, 128, '', 0, 2, 3, models[0], 1, [(256, 0.5), (64, 0.2)], '256x64_gd', 'constant'),
    ]
    
    for (datatype, duration, mels, postfunc, mixup, alpha, argucnt, (mname, cfg), cnt, dense_cfg, dense, padfunc, how) in cfgs:
        #print(duration, mname, mels, cfg)
        use_mfcc = datatype=='mfcc'
        config = Config(sampling_rate=16000, n_mels = mels, 
                        audio_duration=duration, n_folds=nfold, learning_rate=0.001,
                       postfunc=postfunc, mixup=mixup, mixup_alpha = alpha,
                       use_mfcc = use_mfcc)
        if DEBUG:
            config = Config(sampling_rate=16000, n_mels = mels, 
                            audio_duration=duration, n_folds=nfold, max_epochs=1,
                           postfunc=postfunc, mixup=mixup, mixup_alpha = alpha,
                           use_mfcc = use_mfcc)
        config.n_mfcc = config.n_mels
        config.cfg = cfg
        config.data_mname = '{}_{}_{}'.format(datatype, mname, dense)
        print(config.data_mname)

        y_label = np.load('../cache/train_y.npy')
        if DEBUG:
            y_label = y_label[:10]
        y_label = np.argmax(y_label, axis=1)
            
        X_train_original, X_train_mel, X_train_mfcc, y =\
            get_cachedata_all_train(train, train_root, config, argucnt, padfunc)
        X_test_original, X_test_mel, X_test_mfcc =\
            get_cachedata_all_test(test, test_root, config, padfunc)
        
        flex_train = 'train_trim' if 'trim' in train_root else 'train'
        flex_test = 'test_trim' if 'trim' in train_root else 'test'

        
        #wav_statics_train = get_cache_wav_statics(X_train_original, config, flex_train)
        #wav_statics_test = get_cache_wav_statics(X_test_original, config, flex_test)
        #wav_statics_train[wav_statics_train == -np.inf] = 0
        #wav_statics_test[wav_statics_test == -np.inf] = 0
       
        
        #mfcc_statics_train = get_cache_mfcc_statics(X_train_original, config, flex=flex_train)
        #mfcc_statics_test = get_cache_mfcc_statics(X_test_original, config, flex=flex_test)
        
        #mel_statics_train = get_cache_mel_statics(X_train_original, config, flex=flex_train)
        #mel_statics_test = get_cache_mel_statics(X_test_original, config, flex=flex_test)
        
        #seg_statics_train = get_segment_statics(X_train_original, config, flex=flex_train)
        #seg_statics_test = get_segment_statics(X_test_original, config, flex=flex_test)
        
        #train_statics = np.hstack([wav_statics_train, mfcc_statics_train, mel_statics_train, seg_statics_train])
        #test_statics = np.hstack([wav_statics_test, mfcc_statics_test, mel_statics_test, seg_statics_test])
        
        #print('mel', X_train_mel.min(), X_train_mel.max())
        #print('mfcc', X_train_mfcc.min(), X_train_mfcc.max())
        #print('1d', X_train_original.min(), X_train_original.max())
        
        if datatype == 'mel':
            X_train, y = X_train_mel, y
            X_test = X_test_mel   
            
            del X_train_original, X_test_original,X_train_mfcc, X_test_mfcc
            gc.collect()
        elif datatype == 'mfcc':
            X_train = X_train_mfcc
            X_test = X_test_mfcc
            
            del X_train_original, X_test_original,X_train_mel, X_test_mel
            gc.collect()
        elif datatype == 'numeric':
            len_train = len(train_statics)
            all_statics = np.vstack([train_statics, test_statics])
            
            scaler = MinMaxScaler()
            all_statics = scaler.fit_transform(all_statics)
            X_train = all_statics[:len_train]
            X_test = all_statics[len_train:]
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
        
        #X_train, X_test = combine_normalize(X_train, X_test)
        #act_len = len(y_label)  * 1
        #X_train = X_train[:act_len]
        #X_test = X_test[:act_len]
        print('act_len ', X_train.shape, X_test.shape)
            
        if 'pre_' in mname:
            
            print(X_train.shape)
            if 'resnet50' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 197-128),(0, 197-157),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 197-128),(0, 197-157),(0,0)], mode='constant')
            elif 'inception_v3' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 139-128),(0, 0),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 139-128),(0, 0),(0,0)], mode='constant')
            elif 'inception_resnet_v2' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 139-128),(0, 0),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 139-128),(0, 0),(0,0)], mode='constant')
            elif 'mobilenet' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 160-128),(0, 160-157),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 160-128),(0, 160-157),(0,0)], mode='constant')
            elif 'densenet121' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 221-128),(0, 221-157),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 221-128),(0, 221-157),(0,0)], mode='constant')
            elif 'densenet201' in mname:
                X_train = np.pad(X_train, [(0,0),(0, 221-128),(0, 221-157),(0,0)], mode='constant')
                X_test = np.pad(X_train, [(0,0),(0, 221-128),(0, 221-157),(0,0)], mode='constant')
                
            X_train = np.repeat(X_train, 3, axis=-1)
            X_test = np.repeat(X_test, 3, axis=-1)

            
        print('\ntrain shape', X_train.shape)
        print('test shape\n', X_test.shape)
        
        prefix = '{}folds_{}_{}_{}'.format(nfold, duration, mels, config.postfunc)
        if DEBUG:
            prefix += '_debug'
        if mixup > 0:
            prefix += '_mixup_{}_{}_'.format(mixup, config.mixup_alpha)
        if argucnt > 0:
            prefix += 'augu_{}_'.format(argucnt)
        if padfunc != 'constant':
            prefix += '_' + padfunc
        if 'trim' in train_root:
            prefix += '_trim'
            
        PREDICTION_FOLDER = "../result/predictions/{}_{}".format(config.data_mname, prefix)
        if not os.path.exists(PREDICTION_FOLDER):
            os.mkdir(PREDICTION_FOLDER)
            
        config.PREDICTION_FOLDER = PREDICTION_FOLDER
        config.prefix = prefix
        for c in range(cnt):
            if how == 'train':
                oneturn_model(y_label, argucnt, config, X_train, y, X_test, dense_cfg)
            elif how == 'sgd':
                retrain_sgd(y_label, argucnt, config, X_train, y, X_test, dense_cfg)
            
def mix_model_mel_mfcc_1d(LABELS, train, test):
    nfold = 10
    if DEBUG:
        nfold = 2
    batchsize = 24
    
    mels = mfcc = 128
    mixup = 2
    mixup_alpha = 2
    duration = 5
    argucnt = 3
    normalize = False
    config = Config(
            sampling_rate=16000, n_mels = mels, n_mfcc = mfcc,
            audio_duration=duration, n_folds=nfold, learning_rate=0.001,
            postfunc='', mixup=mixup, mixup_alpha = mixup_alpha,
            use_mfcc = True
        )
        
        
    
    #config.n_folds = 2
    #config.max_epochs = 1
    if DEBUG:
        config.max_epochs = 1

    y_label = np.load('../cache/train_y.npy')
    if DEBUG:
        y_label = y_label[:10]
    y_label = np.argmax(y_label, axis=1)
    len_y = len(y_label)

    padfunc = 'constant'
    X_train_original, X_train_mel, X_train_mfcc, y =\
        get_cachedata_all_train(train, train_root, config, argucnt, padfunc)
    X_test_original, X_test_mel, X_test_mfcc =\
        get_cachedata_all_test(test, test_root, config, padfunc)

    X_train_original = X_train_original[:len_y].copy()
    gc.collect()
    X_train_mel = X_train_mel[:len_y].copy()
    gc.collect()
    X_train_mfcc = X_train_mfcc[:len_y].copy()
    gc.collect()
    y = y[:len_y].copy()
    
    gc.collect()
    if normalize is True:
        print('normalize')
        X_train_original, X_test_original = combine_normalize(X_train_original, X_test_original)
        X_train_mel, X_test_mel = combine_normalize(X_train_mel, X_test_mel)
        X_train_mfcc, X_test_mfcc = combine_normalize(X_train_mfcc, X_test_mfcc)
    
    
    X_train_mel = np.repeat(X_train_mel, 3, axis=-1)
    X_test_mel = np.repeat(X_test_mel, 3, axis=-1)
    X_train_mfcc = np.repeat(X_train_mfcc, 3, axis=-1)
    X_test_mfcc = np.repeat(X_test_mfcc, 3, axis=-1)


    print('\ntrain shape', X_train_mel.shape)
    print('test shape\n', X_test_mel.shape)

    cfgs = [
        
        (5, mels, mfcc, 4, 2, 3, 1, [(128, 0.2)], '256-64-noarg', [
                    ('mel', mels, X_train_mel.shape[2], ('pre_xception', []), [(256, 0.5)]),
                    ('mfcc', mfcc, X_train_mfcc.shape[2], ('pre_xception', []), [(256, 0.5)]),
                    ('cnn1d', 0, 0, models[0], [(256, 0.5)]),
                ]),
    ]
    
    for (duration, mels, mfcc, mixup, mixup_alpha, argucnt, cnt, final_dense, flex, multi_cfg) in cfgs:
        config.data_mname = 'mix_{}'.format(flex)
        
        if normalize:
            flex += 'normalize'
        
        X_test_dict = {
            'mel_input_1':X_test_mel,
            'mfcc_input_2':X_test_mfcc,
            'cnn1d':X_test_original,
            #'numeric':X_test,
        }
        """
        X_train = {
            'mel': X_train_mel,
            'mfcc': X_train_mfcc,
            'cnn1d': X_train_original,
        }
        """
        prefix = '{}_{}_{}_{}_{}'.format(config.data_mname, nfold,duration, mels, mfcc)
        if DEBUG:
            prefix += '_debug'
        if mixup > 0:
            prefix += '_mixup_{}_{}_'.format(mixup, config.mixup_alpha)
        PREDICTION_FOLDER = "../result/predictions/{}".format(prefix)
        if not os.path.exists(PREDICTION_FOLDER):
            os.mkdir(PREDICTION_FOLDER)
            
        config.PREDICTION_FOLDER = PREDICTION_FOLDER
        config.prefix = prefix
        
        for c in range(cnt):
            cvscores = []
            
            skf = StratifiedKFold(y_label, n_folds=config.n_folds)
            for i, (train_split, val_split) in enumerate(skf): 
                #train_split = np.hstack([(train_split + len_y * i) for i in range(argucnt -1)])
                print(len(train_split), len(val_split))
                model = make_mixmodel(config, multi_cfg, final_dense)
                #model.summary()
                print('model getted...')
                
                x_valid_dict = {
                    'mel_input_1': X_train_mel[val_split],
                    'mfcc_input_2': X_train_mfcc[val_split],
                    'cnn1d': X_train_original[val_split],
                    #'numeric':X_train[val_split],
                }
                y_valid = y[val_split]
                
                checkpath = '../model/best_{}{}.h5' if not DEBUG else '../model/debug1d_{}{}.h5'
                checkpath = checkpath.format(config.prefix, i)
                    
                if i > 0:
                    print('mixup for fold ', i)
                    mel_ori = X_train_mel[train_split]
                    mfcc_ori = X_train_mfcc[train_split]
                    cnn1d_ori = X_train_original[train_split]
                    #num_ori = X_train[train_split]
                    y_original = y[train_split]
                    if config.mixup > 0:
                        mel_sub = None
                        mfcc_sub = None
                        cnn1d_sub = None
                        #num_sub = None
                        y_train_sub = None
                        #originals = [mel_ori, mfcc_ori, cnn1d_ori, num_ori]
                        originals = [mel_ori, mfcc_ori, cnn1d_ori]
                        for j in range(mixup):
                            #[mel_tmp, mfcc_tmp, cnn1d_tmp, num_tmp], y_tmp = mixup_bulk(originals, y_original, mixup_alpha)
                            [mel_tmp, mfcc_tmp, cnn1d_tmp], y_tmp = mixup_bulk(originals, y_original, mixup_alpha)
                            
                            y_train_sub = y_tmp if y_train_sub is None else np.vstack((y_train_sub, y_tmp))
                            del y_tmp; gc.collect()
                            
                            mel_sub = mel_tmp if mel_sub is None else np.vstack((mel_sub, mel_tmp))
                            del mel_tmp; gc.collect()
                            mfcc_sub = mfcc_tmp if mfcc_sub is None else np.vstack((mfcc_sub, mfcc_tmp))
                            del mfcc_tmp; gc.collect()
                            
                            cnn1d_sub = cnn1d_tmp if cnn1d_sub is None else np.vstack((cnn1d_sub, cnn1d_tmp))
                            del cnn1d_tmp; gc.collect()
                            
                            #num_sub = num_tmp if num_sub is None else np.vstack((num_sub, num_tmp))
                            #del num_tmp; gc.collect()
                            
                            
                    else:
                        y_train_sub = y_original
                        mel_sub = mel_ori
                        mfcc_sub = mfcc_ori
                        cnn1d_sub = cnn1d_ori
                        #num_sub = num_ori
                        

                    x_train_sub_dict = {
                        'mel_input_1': mel_sub,
                        'mfcc_input_2': mfcc_sub,
                        'cnn1d': cnn1d_sub,
                        #'numeric':num_sub,
                    }
                
                    
                    checkpoint = ModelCheckpoint(checkpath, monitor='val_acc', verbose=1, save_best_only=True)
                    early = EarlyStopping(monitor="val_acc", patience=5)
                    
                    callbacks_list = [checkpoint, early]
                    if not DEBUG:
                        tensorboard = TensorBoard(log_dir='./graph/{}_normalize'.format(config.prefix), histogram_freq=0,  write_graph=True, write_images=True)
                        callbacks_list.append(tensorboard)

                    model.fit(x_train_sub_dict, y_train_sub, batch_size=batchsize, epochs=config.max_epochs, 
                                validation_data=(x_valid_dict, y_valid), verbose=1,
                                callbacks=callbacks_list)
                else:
                    print('skip train fold', i)
                    
                model.load_weights(checkpath)
                #model = keras.models.load_model(checkpath)

                scores = model.evaluate(x_valid_dict, y_valid)
                logger.info('{}_{} val_loss: {:.4f} val_acc: {:.4f}'.format(
                    config.data_mname,config.prefix, scores[0], scores[1]))
                print(scores[1])
                cvscores.append(scores[1])
                # Save train predictions
                print('save result npy')
                predictions = model.predict(X_test_dict, batch_size = batchsize, verbose = 1)
                savepath = "/p{}.npy"
                savepath = savepath.format(i)
                np.save(config.PREDICTION_FOLDER + savepath, predictions)
                
                #train_predicts = model.predict(X_train, batch_size = batchsize, verbose = 1)
                #savepath = "/p_train{}_{}.npy".format('mix_all', i)
                #np.save(config.PREDICTION_FOLDER + savepath, train_predicts)
                
                K.clear_session()
                #break 
            
            cvmean = np.mean(cvscores)
            cvstd = np.std(cvscores)
            print('mean {0:.3f} std {1:.3f}'.format(cvmean, cvstd))
            actual_prefix = config.prefix + '_{:.3f}_{:.3f}'.format(cvmean, cvstd)
            ensemble(LABELS, config.n_folds, [config.PREDICTION_FOLDER], actual_prefix, 'mix', False)
            
            
if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv", index_col="fname")
    test = pd.read_csv("../data/sample_submission.csv", index_col="fname")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    if DEBUG:
        train = train[:100]
        test = test[:100]
    
    single_models(LABELS, train, test)
    #mix_model_mel_mfcc_1d(LABELS, train, test)
    #mix_model_mel_mfcc(LABELS, train, test)
    
