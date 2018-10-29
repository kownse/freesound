import librosa
import numpy as np
import pandas as pd
import scipy
import os
from keras.utils import Sequence, to_categorical
from tqdm import tqdm
import kaggle_util

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=200, n_mfcc=20, n_mels = 32, postfunc='', mixup = 0, mixup_alpha = 0.5):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.postfunc=postfunc
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.cfg = None
        self.data_mname = ''
        self.PREDICTION_FOLDER = ''
        self.prefix = ''
        self.opt = 'adam'

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mels, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)
            
class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

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

    def __data_generation(self, list_IDs_temp, progress=False):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim), dtype=np.float32)

        input_length = self.config.audio_length
        if progress:
            pbar = tqdm(total=cur_batch_size)
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            try:
                # Read and Resample the audio
                data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                            res_type='kaiser_fast')
            except:
                file_path = '../data/audio_test/' + ID
                data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                            res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "wrap")
                
            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                                   n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data.astype(np.float32)
            
            if progress:
                pbar.update(1)
            
        if progress:
            pbar.close()

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes).astype(np.uint8)
        else:
            return X

def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    pbar = tqdm(total=df.shape[0])
    for i, fname in enumerate(df.index):
        #print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "wrap")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)

        X[i,] = data
        pbar.update(1)
        
    pbar.close()
    return X

def getCacheMFCC(df, config, data_dir, flex=''):
    path =  '../cache/mfcc_{}_{}_{}_{}_{}.npy'.format(flex, 
                                                      config.sampling_rate, 
                                                      config.audio_duration, 
                                                      config.n_mfcc,
                                                      len(df))
    
    if os.path.exists(path):
        mfcc = np.load(path)
    else:
        print(path, 'not exist')
        mfcc = prepare_data(df, config, data_dir)
        np.save(path, mfcc)
        
    return mfcc

def getCacheTrainData(train, path, sr = 100, duration = 1, flex = ''):
    train_x_path = '../cache/cache_train_x_{}_{}_{}{}.npy'.format(sr, duration, len(train), flex)
    train_y_path = '../cache/cache_train_y_{}_{}_{}{}.npy'.format(sr, duration, len(train), flex)
    
    
    if os.path.exists(train_x_path) and os.path.exists(train_y_path):
        X_train = np.load(train_x_path)
        y = np.load(train_y_path)
    else:
        print(train_x_path, 'not exist')
        config = Config(sampling_rate=sr, audio_duration=duration)
        train_generator = DataGenerator(config, path, train.index, 
                                            train.label_idx, batch_size=64,
                                            preprocessing_fn=audio_norm)
        X_train,y = train_generator.getitem_all()
        np.save(train_x_path, X_train)
        np.save(train_y_path, y)
        
    return X_train, y

def getCacheTestData(test, path, sr = 100, duration = 1, flex = ''):
    test_x_path = '../cache/cache_test_x_{}_{}_{}{}.npy'.format(sr, duration, len(test), flex)
    if os.path.exists(test_x_path):
        X_test = np.load(test_x_path)
    else:
        print(test_x_path, 'not exist')
        config = Config(sampling_rate=sr, audio_duration=duration)
        test_generator = DataGenerator(config, path, test.index, 
                                            batch_size=64,
                                            preprocessing_fn=audio_norm)
        X_test = test_generator.getitem_all()
        np.save(test_x_path, X_test)
    return X_test



def calc_ensemble(pred_list):
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))
    return prediction

def save_ensemble(LABELS, prediction, prefix, mname, send):
    np.save('../result/ensembles/{}_{}.npy'.format(mname, prefix), prediction)
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('../data/sample_submission.csv')
    test['label'] = predicted_labels
    
    filename = '../result/{}_{}.csv'.format(mname, prefix)
    kaggle_util.save_result(test[['fname', 'label']],
                           filename,
                           'freesound-audio-tagging',
                           send = send, index=False)

def ensemble_list(LABELS, pred_list, prefix, mname, send):
    prediction = calc_ensemble(pred_list)
    save_ensemble(LABELS, prediction, prefix, mname, send)
    
def ensemble(LABELS, nfold, root_paths, prefix, mname, send):
    print('ensemble...')
    
    df_tot = pd.DataFrame()
    pred_list = []
    cnt = 0
    for root_path in root_paths:
        sub_prelist = []
        for i in range(nfold):
            savepath = "/p{}.npy"
            savepath = savepath.format(i)
            pred = np.load(root_path + savepath)
            pred_list.append(pred)
            sub_prelist.append(pred)
            
        sub_pred = calc_ensemble(sub_prelist)
        df_tot['sub{}'.format(cnt)] = sub_pred.reshape(-1)
        cnt += 1
    print(df_tot.corr())
    ensemble_list(LABELS, pred_list, prefix, mname, send)
    
def ensemble_results(LABELS, paths, prefix, mname, send):
    df_tot = pd.DataFrame()
    pred_list = []
    cnt = 0
    for (name, path, ratio) in tqdm(paths):
        savepath = '../result/ensembles/{}.npy'.format(path)
        pred = np.load(savepath)
        df_tot[name] = pred.reshape(-1)
        pred = pred ** ratio
        pred_list.append(pred)
        cnt += 1
    print(df_tot.corr())
    ensemble_list(LABELS, pred_list, prefix, mname, send)
    
def ensemble_results_ratio(LABELS, paths, prefix, mname, send):
    df_tot = pd.DataFrame()
    result = None
    for (name, path, ratio) in tqdm(paths):
        savepath = '../result/ensembles/{}.npy'.format(path)
        pred = np.load(savepath)
        df_tot[name] = pred.reshape(-1)
        pred *= ratio
        
        if result is None:
            result = pred
        else:
            result += pred
    print(df_tot.corr())
    save_ensemble(LABELS, result, prefix, mname, send)
    
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def get_valid_score(y_valid, p):
    predictions = [list(np.argsort(p[i])[::-1][:3]) for i in range(len(p))]
    actual = [[i] for i in y_valid]
    valid_score = mapk(actual, predictions, k=3)
    return valid_score