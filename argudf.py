import numpy as np
np.random.seed(1001)
import os
import shutil
import pandas as pd
from util import *
from dataset import *
from cnn2d import *
from transform import *
import librosa

def argu_df(df, root, idx):
    callbacks = [
        LoadAudio(None),
        ChangeAmplitude(), 
        ChangeSpeedAndPitchAudio(), 
        StretchAudio(),
        TimeshiftAudio(),
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

train_root = '../data/audio_train/'
test_root = '../data/audio_test/'

nrows = None
test_df = pd.read_csv("../data/sample_submission.csv", nrows = nrows)
train_df = pd.read_csv("../data/train.csv", nrows = nrows)

