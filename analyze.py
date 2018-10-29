import os, random, math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb

import librosa
import librosa.display

import scipy
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
#from tqdm import tqdm_notebook, 
from tqdm import tqdm_pandas, tqdm
#tqdm_notebook().pandas(smoothing=0.7)
tqdm.pandas(desc="my bar!")

from tqdm import tqdm
from kaggle_util import reduce_mem_usage
from dataset import *

def clean_np(data):
    data[data == np.inf] = 0
    data[np.isnan(data)] = 0
    return data

def wavfile_stats(fname, root):
    try:
        data, fs = librosa.core.load(root + fname, sr=None)
        mean = np.mean(data)
        minimum = np.min(data)
        maximum = np.max(data)
        std = np.std(data)
        length = len(data)
        rms = np.sqrt(np.mean(data**2))
        skewness = skew(data)
        kurt = kurtosis(data)

        return pd.Series([length, mean, minimum, maximum, std, rms, skewness, kurt])
    except ValueError:
        print("Bad file at {}".format(fname))
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
    
def trim_silence(fname, root, window_length=0.5):
    try:
        trimmed_ends = 0
        trimmed_int = 0
        
        data, fs = librosa.core.load(root + fname, sr=None)
        length = len(data) / 16000
        
        # Trim silence from ends
        limit = 40
        data, _ = librosa.effects.trim(data, top_db=limit)
        length_int = len(data) / 16000
        ratio_int = length_int/length
        
        # Split file into non-silent chunks and recombine   
    except ValueError:
        print("Bad file at {}".format(fname))
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0])  
    
    splits = librosa.effects.split(data, top_db=40)
    cnt_splits = len(splits)
    split_data = [data[x[0]:x[1]] for x in splits if x[1] > x[0]]
    #print(fname, len(split_data))
    split_max = [np.max(sub) for sub in split_data]
    mean_max_splits = np.mean(split_max)
    std_max_splits = np.std(split_max)

    lens = [(x[1] - x[0]) / 16000 for x in splits]
    mean_len_splits = np.mean(lens)
    std_len_splits = np.std(lens)

    length_final = np.sum(lens) 
    ratio_final = length_final /length_int     

    # Save file and return new features
    librosa.output.write_wav('{}_trimmed/{}'.format(root[:-1], fname), data, fs)
    return pd.Series([length_int, length_final, ratio_int, ratio_final, mean_max_splits, std_max_splits, mean_len_splits, std_len_splits, cnt_splits])
    
def spectral_features(fname=None, root=None, n_mfcc=20, return_fnames=False):
    feature_names = []
    for i in ['mean', 'std', 'min', 'max', 'skew', 'kurt']:
        for j in range(n_mfcc):
            feature_names.append('mfcc_{}_{}'.format(j, i))
        feature_names.append('centroid_{}'.format(i))
        feature_names.append('bandwidth_{}'.format(i))
        feature_names.append('contrast_{}'.format(i))
        feature_names.append('rolloff_{}'.format(i))
        feature_names.append('flatness_{}'.format(i))
        feature_names.append('zcr_{}'.format(i))
    
    if return_fnames:
        return feature_names

    spectral_features = [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_contrast,
        librosa.feature.spectral_rolloff,
        librosa.feature.spectral_flatness,
        librosa.feature.zero_crossing_rate]
     
    try:
        data, fs = librosa.core.load(root + fname, sr=None)
        M = librosa.feature.mfcc(data, sr=fs, n_mfcc=n_mfcc)
        data_row = np.hstack((np.mean(M, axis=1), np.std(M, axis=1), np.min(M, axis=1),
                              np.max(M, axis=1), skew(M, axis=1), kurtosis(M, axis=1)))
        
        for feat in spectral_features:
            S = feat(data)[0]
            data_row = np.hstack((data_row, np.mean(S), np.std(S), np.min(S),
                                  np.max(S), skew(S), kurtosis(S)))

        return pd.Series(data_row)
        
    except:
        print("Bad file at {}".format(fname))
        return pd.Series([0]*len(feature_names)) 
    
def mel_spectral_features(fname=None, root=None, n_mels=32, return_fnames=False):
    feature_names = []
    for i in ['mean', 'std', 'min', 'max', 'skew', 'kurt']:
        for j in range(n_mels):
            feature_names.append('mel_{}_{}'.format(j, i))
    
    if return_fnames:
        return feature_names
    
    try:
        data, fs = librosa.core.load(root + fname, sr=None)
        n_fft = 2048
        stft = librosa.stft(data, n_fft=n_fft, hop_length=512)
        mel_basis = librosa.filters.mel(fs, n_fft, n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        M = librosa.power_to_db(s, ref=np.max)
        
        data_row = np.hstack((np.mean(M, axis=1), np.std(M, axis=1), np.min(M, axis=1),
                              np.max(M, axis=1), skew(M, axis=1), kurtosis(M, axis=1)))
        
        return pd.Series(data_row)
        
    except:
        print("Bad file at {}".format(fname))
        return pd.Series([0]*len(feature_names)) 
    
def basic_analyze(train_df, test_df, train_root, test_root):
    print('basic_analyze')
    train_df[['length', 'data_mean', 'data_min', 'data_max', 'data_std', 'data_rms', 'skewness', 'kurtosis']] = train_df['fname'].progress_apply(wavfile_stats, root=train_root)
    test_df[['length', 'data_mean', 'data_min', 'data_max', 'data_std', 'data_rms', 'skewness', 'kurtosis']] = test_df['fname'].progress_apply(wavfile_stats, root=test_root)
    
    train_df['rms_std'] = train_df['data_rms'] / train_df['data_std']
    test_df['rms_std'] = test_df['data_rms'] / test_df['data_std']

    train_df['max_min'] = train_df['data_max'] / train_df['data_min']
    test_df['max_min'] = test_df['data_max'] / test_df['data_min']
    
    train_df.to_csv('../data/train_1.csv', index=False, float_format='%.4f')
    test_df.to_csv('../data/test_1.csv', index=False, float_format='%.4f')
    
    return reduce_mem_usage(train_df), reduce_mem_usage(test_df)

def analyze_mfcc(train_df, test_df):
    print('analyze_mfcc')
    feature_names = spectral_features(return_fnames=True)
    train_df[feature_names] = train_df['fname'].progress_apply(spectral_features, root=train_root_trimmed)
    test_df[feature_names] = test_df['fname'].progress_apply(spectral_features, root=test_root_trimmed)

    train_df.to_csv('../data/train_spectral.csv', index=False, float_format='%.4f')
    test_df.to_csv('../data/test_spectral.csv', index=False, float_format='%.4f')
    
    return reduce_mem_usage(train_df), reduce_mem_usage(test_df)

def analyze_mel(train_df, test_df):
    print('analyze_mel')
    feature_names = mel_spectral_features(return_fnames=True)
    train_df[feature_names] = train_df['fname'].progress_apply(mel_spectral_features, root=train_root_trimmed)
    test_df[feature_names] = test_df['fname'].progress_apply(mel_spectral_features, root=test_root_trimmed)

    train_df.to_csv('../data/train_mel.csv', index=False, float_format='%.4f')
    test_df.to_csv('../data/test_mel.csv', index=False, float_format='%.4f')
    
    return reduce_mem_usage(train_df), reduce_mem_usage(test_df)

def trim(train_df, test_df):
    print('trim')
    train_df[['length_int', 'length_final', 'ratio_int', 'ratio_final', 'mean_max_splits', 'std_max_splits', 'mean_len_splits', 'std_len_splits', 'cnt_splits']] = \
        train_df['fname'].progress_apply(trim_silence, root=train_root)
    
    train_df.to_csv('../data/train_2.csv', index=False, float_format='%.4f')
    
    test_df[['length_int', 'length_final', 'ratio_int', 'ratio_final', 'mean_max_splits', 'std_max_splits', 'mean_len_splits', 'std_len_splits', 'cnt_splits']] = \
        test_df['fname'].progress_apply(trim_silence, root=test_root)
    
    test_df.to_csv('../data/test_2.csv', index=False, float_format='%.4f')
    
    return reduce_mem_usage(train_df), reduce_mem_usage(test_df)

def extract_features(files, path):
    print('extract_features')
    features = {}

    cnt = 0
    for f in tqdm(files):
        try:
            fs, data = scipy.io.wavfile.read(os.path.join(path, f))
        except:
            continue
            
        features[f] = {}
        
        def calc_part_features(data, n=2, prefix=''):
            f_i = 1
            for i in range(0, len(data), len(data)//n):
                features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + len(data)//n])
                features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + len(data)//n])
                features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + len(data)//n])
                features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + len(data)//n])
        
        abs_data = np.abs(data)
        diff_data = np.diff(data)

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
    
    return features

def extract_segment_feature(train, test):
    train_files = train.fname.values
    train_features = extract_features(train_files, train_root_trimmed)
    
    test_files = test.fname.values
    test_features = extract_features(test_files, test_root_trimmed)
    
    train = train.merge(train_features, on='fname', how='left')
    test = test.merge(test_features, on='fname', how='left')
    
    train.to_csv('../data/train_seg.csv', index=False, float_format='%.4f')
    test.to_csv('../data/test_seg.csv', index=False, float_format='%.4f')
    
    return reduce_mem_usage(train), reduce_mem_usage(test)

def argu_df(df, root, idx):
    callbacks = [
        LoadAudio(None),
        ChangeAmplitude(), 
        ChangeSpeedAndPitchAudio(), 
        StretchAudio(),
        TimeshiftAudio(),
        RandomAudioLength(),
    ]
    
    df_argu = pd.DataFrame()
    pbar = tqdm(total=len(df))
    for row in df.iterrows():
        file_path = train_root + row[1].fname
        data =  prepare_wav(file_path, callbacks)
        ID = row[1].fname[:row[1].fname.find('.')]
        save_file = '{}_arg{}.wav'.format(ID, idx)
        save_path = root + save_file
        #print(save_path)
        librosa.output.write_wav(save_path, data['samples'], data['sample_rate'])
        
        new_row = row[1].copy()
        new_row.fname = save_file
        #print()
        df_argu = df_argu.append(new_row, ignore_index=True)
        
        pbar.update(1)
    pbar.close()
    
    return df_argu

def argu_df_bulk(df, root, cnt):
    print('argu df')
    for i in tqdm(range(cnt)):
        df_argu = argu_df(train_df, train_root, i)
        df = df.append(df_argu, ignore_index=True)
        
    return df
    
def wav_stats(wav):
    from scipy.stats import skew, kurtosis
    analyzes = [
        np.mean,
        np.min,
        np.max,
        np.std,
        lambda x : len(x) / 16000,
        lambda x : np.sqrt(np.mean(x**2)),
        skew,
        kurtosis,
    ]
    
    cnt = len(wav)
    res = np.zeros((cnt, 8), np.float32)
    for i in tqdm(range(cnt)):
        data = wav[i]
        for j, callback in enumerate(analyzes):
            res[i,j] = 0 if len(data) <= 0 else callback(data)
    return clean_np(res)

def get_cache_wav_statics(wav, config, flex = ''):
    flex = '{}_{}_{}_{}'.format(config.sampling_rate, config.audio_duration,
                                len(wav), flex)
    statics_path = '../cache/wav_statics_{}.npy'.format(flex)
    if os.path.exists(statics_path):
        wav_statics = np.load(statics_path)
    else:
        wav_statics = wav_stats(wav)
        np.save(statics_path, wav_statics)
    return wav_statics

def mfcc_stats_one(data):
    M = librosa.feature.mfcc(data, sr=16000, n_mfcc=20)
    data_row = np.hstack((np.mean(M, axis=1), np.std(M, axis=1), np.min(M, axis=1),
                              np.max(M, axis=1), skew(M, axis=1), kurtosis(M, axis=1)))
    spectral_features = [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_contrast,
        librosa.feature.spectral_rolloff,
        librosa.feature.spectral_flatness,
        librosa.feature.zero_crossing_rate
    ]
    for feat in spectral_features:
        S = feat(data)[0]
        #print(data_row.shape, S.shape)
        data_row = np.hstack((data_row, np.mean(S), np.std(S), np.min(S),
                              np.max(S), skew(S), kurtosis(S)))
        
    return data_row

def mfcc_stats(wav):
    res = np.zeros((len(wav), 156), np.float32)
    for i in tqdm(range(len(wav))):
        data = wav[i].squeeze()
        data_row = mfcc_stats_one(data)
        res[i] = data_row
    return clean_np(res)

def get_cache_mfcc_statics(wav, config, flex = ''):
    flex = '{}_{}_{}_{}'.format(config.sampling_rate, config.audio_duration,
                                len(wav), flex)
    statics_path = '../cache/mfcc_statics_{}.npy'.format(flex)
    if os.path.exists(statics_path):
        mfcc_statics = np.load(statics_path)
    else:
        mfcc_statics = mfcc_stats(wav)
        np.save(statics_path, mfcc_statics)
    return mfcc_statics

def mel_stats_one(data):
    n_fft = 2048
    stft = librosa.stft(data, n_fft=n_fft, hop_length=512)
    mel_basis = librosa.filters.mel(16000, n_fft, 32)
    s = np.dot(mel_basis, np.abs(stft)**2.0)
    M = librosa.power_to_db(s, ref=np.max)
    data_row = np.hstack((np.mean(M, axis=1), np.std(M, axis=1), np.min(M, axis=1),
                              np.max(M, axis=1), skew(M, axis=1), kurtosis(M, axis=1)))
        
    return data_row

def mel_stats(data):
    res = np.zeros((len(data), 192), np.float32)
    for i in tqdm(range(len(data))):
        subdata = data[i].squeeze()
        data_row = mel_stats_one(subdata)
        res[i] = data_row
    return clean_np(res)

def get_cache_mel_statics(data, config, flex = ''):
    flex = '{}_{}_{}_{}'.format(config.sampling_rate, config.audio_duration,
                                len(data), flex)
    statics_path = '../cache/mel_statics_{}.npy'.format(flex)
    if os.path.exists(statics_path):
        mel_statics = np.load(statics_path)
    else:
        mel_statics = mel_stats(data)
        np.save(statics_path, mel_statics)
    return mel_statics

def segment_stats(datas):
    features = {}
    
    for f in tqdm(range(len(datas))):
        data = datas[f].squeeze()
        features[f] = {}
        def calc_part_features(data, n=2, prefix=''):
            f_i = 1
            for i in range(0, len(data), len(data)//n):
                subdata = data[i:i + len(data)//n]
                features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(subdata)
                features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(subdata)
                features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(subdata)
                features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(subdata)
        abs_data = np.abs(data)
        diff_data = np.diff(data)
        #print(abs_data.shape,diff_data.shape)
        features[f]['len'] = len(data) / 16000
        if features[f]['len'] > 0:
            #print(features[f]['len'])
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
    
    features = pd.DataFrame(features).T.reset_index()
    return clean_np(features.drop('index', axis=1).values.astype(np.float32))

def get_segment_statics(wav, config, flex = ''):
    flex = '{}_{}_{}_{}'.format(config.sampling_rate, config.audio_duration,
                                len(wav), flex)
    statics_path = '../cache/segment_statics_{}.npy'.format(flex)
    if os.path.exists(statics_path):
        seg_statics = np.load(statics_path)
    else:
        seg_statics = segment_stats(wav)
        np.save(statics_path, seg_statics)
    return seg_statics

if __name__ == "__main__":
    train_root = '../data/audio_train/'
    test_root = '../data/audio_test/'
    
    #train_root_trimmed = train_root
    #test_root_trimmed = test_root
    
    train_root_trimmed = '../data/audio_train_trimmed/'
    test_root_trimmed = '../data/audio_test_trimmed/'

    os.makedirs('../data/audio_train_trimmed', exist_ok=True)
    os.makedirs('../data/audio_test_trimmed', exist_ok=True)
    
    nrows = None
    #test_df = pd.read_csv("../data/sample_submission.csv", nrows = nrows)
    #train_df = pd.read_csv("../data/train.csv", nrows = nrows)
    test_df = pd.read_csv("../data/test_1.csv", nrows = nrows)
    train_df = pd.read_csv("../data/train_1.csv", nrows = nrows)
    
    df_argu_train = argu_df_bulk(train_df, train_root, 1)
    df_argu_train.to_csv('../data/train_argu.csv', index=False)
    
    train_df = df_argu_train
    
    #train_df, test_df = basic_analyze(train_df, test_df, train_root, test_root)
    #train_df, test_df = trim(train_df, test_df)
    train_df, test_df = analyze_mfcc(train_df, test_df)
    train_df, test_df = extract_segment_feature(train_df, test_df)
    train_df, test_df = analyze_mel(train_df, test_df)