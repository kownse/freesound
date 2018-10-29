#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:32:42 2018

@author: kownse
"""

import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import gc

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm(df.columns):
        col_type = df[col].dtype
        #print(col_type)
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if 'int' in str(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def replaceCatWithDummy(df, catname):
    dummies = pd.get_dummies(df[catname], prefix=catname)
    df = df.drop(catname, axis=1)
    gc.collect()
    return pd.concat([df, dummies], axis=1)

def replaceCatWithDummy_bulk(df, cols):
    for col in cols:
        df = replaceCatWithDummy(df, col)
        gc.collect()
    return df

def save_result(sub, filename, competition = '', send = False, index = False):
    print('save result')
    sub.to_csv(filename, index=index)
    
    if send and len(competition) > 0:
        print('zip result')
        file_7z = '{}.7z'.format(filename)
        os.system('7z a {} {}'.format(file_7z, filename))
        
        print('upload result')
        command = '/home/kownse/anaconda3/bin/kaggle competitions submit -c {} -f {} -m "submit"'.format(competition, file_7z)
        print('cmd: ' + command)
        os.system(command)
        
def read_result(path, idx, score_col = 'deal_probability'):
    compression = None
    if '.gz' in path:
        compression='gzip'
    return pd.read_csv('../result/' + path, compression=compression).rename(columns={score_col: 'p{}'.format(idx)})
    
def ensemble(result_list, send, competition = '', score_col = 'deal_probability', prefix = 'ensemble'):
    print('score_col ', score_col)
    sub = read_result(result_list[0][0], 0, score_col = score_col)
    sub['p0'] *= result_list[0][1]
    for i in tqdm(range(1, len(result_list))):
        res = read_result(result_list[i][0], 0, score_col = score_col)
        sub['p{}'.format(i)] = res['p0'] * result_list[i][1]
    
    print(sub.corr())
    #return

    sub[score_col] = 0 #(sub['p0'] + sub['p1'] + sub['p2']) / 3
    for i in tqdm(range(len(result_list))):
        sub[score_col] += sub['p{}'.format(i)]
        sub.drop('p{}'.format(i), axis = 1, inplace = True)

    str_now = datetime.now().strftime("%m-%d-%H-%M")
    filename = '../result/{}_{}.csv'.format(prefix, str_now)
    save_result(sub, filename, competition = competition, send = send)
    
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def get_text_tokenizer(df, col, max_words):
    import keras.preprocessing
    
    tokenizer = keras.preprocessing.text.Tokenizer(num_words = max_words)
    alltext = np.hstack([df[col].str.lower()])
    tokenizer.fit_on_texts(alltext)
    
    del alltext
    gc.collect()
    return tokenizer

def build_emb_matrix_from_tokenizer(tokenizer, path, EMBEDDING_DIM1):
    EMBEDDING_FILE1 = path
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))
    
    vocab_size = len(tokenizer.word_index)+2
    embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
    print(embedding_matrix1.shape)
    # Creating Embedding matrix 
    c = 0 
    c1 = 0 
    w_Y = []
    w_No = []
    for word, i in tokenizer.word_index.items():
        if word in embeddings_index1:
            c +=1
            embedding_vector = embeddings_index1[word]
            w_Y.append(word)
        else:
            embedding_vector = None
            w_No.append(word)
            c1 +=1
        if embedding_vector is not None:    
            embedding_matrix1[i] = embedding_vector

    print(c,c1, len(w_No), len(w_Y))
    print(embedding_matrix1.shape)
    del embeddings_index1
    gc.collect()
    
    return embedding_matrix1, vocab_size

def build_emb_matrix_from_w2v(path):
    EMBEDDING_FILE1 = path
    model = Word2Vec.load(EMBEDDING_FILE1)
    word_index = tokenizer.word_index
    vocab_size = min(max_words_title_description, len(word_index))
    embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
    for word, i in word_index.items():
        if i >= max_words_title_description: continue
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None: embedding_matrix1[i] = embedding_vector

    return embedding_matrix1, vocab_size