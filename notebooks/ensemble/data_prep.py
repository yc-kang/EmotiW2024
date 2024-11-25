# There are 3 dataloaders, data_loader_fusion, data_loader_v1, data_loader_triple

import os
import glob
import utils
import config

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler() 

n_segments = config.N_SEGMENTS

# Note that base_dir='EmotiW2023 Data Small' just for testing purposes

def pad_sequences(sequences):
    # Find max length and feature dimension
    max_length = max(len(seq) for seq in sequences)
    feature_dim = sequences[0].shape[1]
    
    # Initialize array for padded sequences
    padded_sequences = np.zeros((len(sequences), max_length, feature_dim))
    
    # Fill the array with actual sequences
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq), :] = seq
        
    return padded_sequences

def data_loader_triple(feature_type, val=True, base_dir='EmotiW2023 Data Small'):
    labels = pd.read_excel(f'{base_dir}/engagement_labels.xlsx', index_col=0)
    print(labels.head())

    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True)
    print(type(Xy), len(Xy))
    
    features_label_map = {}
    for xy in Xy:
        features_label_map[xy[0]] = (xy[1], xy[2], xy[3], xy[4])
        
    # Initialize arrays
    train_x_1, train_x_2, train_x_3 = [], [], []
    train_y = []
    
    if val:
        val_x_1, val_x_2, val_x_3 = [], [], []
        val_y = []
        
    test_x_1, test_x_2, test_x_3 = [], [], []
    test_y = []

    train_not_found_count = 0
    val_not_found_count = 0
    test_not_found_count = 0
    
    # Load split files
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')

    # Testing
    print('TrainXy:', trainXy[:5])
    print('len TrainXy:', len(trainXy))
    print('TestXy:', testXy[:5])
    print('len TestXy:', len(testXy))
    print('ValXy:', valXy[:5])
    print('len ValXy:', len(valXy))

     # Training loop
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[3] != config.SNP:
                train_x_1.append(xy[0])
                train_x_2.append(xy[1])
                train_x_3.append(xy[2])
                train_y.append(config.LABEL_MAP[xy[3]])
        except KeyError as k:
            print ('not found(train): ', k)
            train_not_found_count += 1
            # pass

    # Need to implement some sort of data scaling (But not sure if it's openface or marlin)
    # Important!!!
    train_x_3_padded = pad_sequences(train_x_3)
    
    # Reshape for scaling
    original_shape = train_x_3_padded.shape
    X = train_x_3_padded.reshape(-1, train_x_3_padded.shape[-1])
    
    # Fit and transform
    scaler.fit(X)
    X = scaler.transform(X)
    
    # Reshape back
    train_x_3_scaled = X.reshape(original_shape)

    # Validation loop
    if val:
        for e in valXy:
            try:
                xy = features_label_map[e]
                if xy[3] != config.SNP:
                    val_x_1.append(xy[0])
                    val_x_2.append(xy[1])
                    val_x_3.append(xy[2])
                    val_y.append(config.LABEL_MAP[xy[3]])
            except KeyError as k:
                print('not found(val): ', k)
                val_not_found_count += 1
        
        # Process validation data
        val_x_3_padded = pad_sequences(val_x_3)
        val_shape = val_x_3_padded.shape
        val_x_3_scaled = scaler.transform(val_x_3_padded.reshape(-1, val_x_3_padded.shape[-1])).reshape(val_shape)

    
    # Validation loop
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[3] != config.SNP:
                x = xy[2]
                x = scaler.transform(xy[2])
                
                if val:
                    val_x_3.append(x)
                    val_x_2.append(xy[1])
                    val_x_1.append(xy[0])
                    val_y.append(config.LABEL_MAP[xy[3]])
                else:
                    train_x_1.append(xy[0])
                    train_x_2.append(xy[1])
                    train_x_3.append(x)
                    train_y.append(config.LABEL_MAP[xy[3]])
        except KeyError as k:
            print ('not found(val): ', k)
            val_not_found_count += 1
    
    # Test loop
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[3] != config.SNP:
                x = xy[2]
                x = scaler.transform(xy[2])
                
                test_x_1.append(xy[0])
                test_x_2.append(xy[1])
                test_x_3.append(x)
                test_y.append(config.LABEL_MAP[xy[3]])
        except KeyError as k:
            print ('not found(test): ', k)
            test_not_found_count += 1
        
    if val:
        return (
            ((np.array(train_x_1), np.array(train_x_2), np.array(train_x_3)), np.array(train_y)),
            ((np.array(val_x_1), np.array(val_x_2), np.array(val_x_3)), np.array(val_y)),
            ((np.array(test_x_1), np.array(test_x_2), np.array(test_x_3)), np.array(test_y))
        )
    else:
        return (
            ((np.array(train_x_1), np.array(train_x_2), np.array(train_x_3)), np.array(train_y)),
            ((np.array(test_x_1), np.array(test_x_2), np.array(test_x_3)), np.array(test_y))
        )

def data_loader_fusion(feature_type, val=True, base_dir='EmotiW2023 Data Small'):
    labels = pd.read_excel(f'{base_dir}/engagement_labels.xlsx', index_col=0) # Load label file
    print(labels.head())
    
    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True)
    print(type(Xy), len(Xy))
    
    features_label_map = {}
    for xy in Xy:  
        features_label_map[xy[0]] = (xy[1], xy[2], xy[3])
    
    train_x_1 = []
    train_x_2 = []
    train_y = []
    if val:
        val_x_1 = []
        val_x_2 = []
        val_y = []
        
    test_x_1 = []
    test_x_2 = []
    test_y = []

    train_not_found_count = 0
    val_not_found_count = 0
    test_not_found_count = 0
    
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')
    print('TrainXy:', trainXy[:5])
    print('len TrainXy:', len(trainXy))
    print('TestXy:', testXy[:5])
    print('len TestXy:', len(testXy))
    print('ValXy:', valXy[:5])
    print('len ValXy:', len(valXy))

    # Training loop
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
                train_x_1.append(xy[0])
                train_x_2.append(xy[1])
                train_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            print ('not found(train): ', k)
            train_not_found_count += 1
            # pass

    # Apply scaling
    X = np.array(train_x_2)
    scaler.fit(X.reshape(-1, X.shape[-1]))
    for i in range(len(train_x_2)):
        train_x_2[i] = scaler.transform(train_x_2[i])

    # Validation loop
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
#                     
                x = xy[1]
                
                x = scaler.transform(xy[1])
                if val:
                    val_x_2.append(x)
                    val_x_1.append(xy[0])
                    val_y.append(config.LABEL_MAP[xy[2]])
                else:
                    train_x_1.append(xy[0])
                    train_x_2.append(x)
                    train_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            print ('not found(val): ', k)
            val_not_found_count += 1
            # pass
#                 print ('not found(val): ', k)

    # Test loop
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
                
                x = xy[1]
                
                x = scaler.transform(xy[1])
                
                test_x_1.append(xy[0])
                test_x_2.append(x)
                test_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            print ('not found(test): ', k)
            test_not_found_count += 1
            # pass
#             print ('not found(test): ', k)
    if val:
        return (
            ((np.array(train_x_1), np.array(train_x_2)), np.array(train_y)), 
            ((np.array(val_x_1), np.array(val_x_2)), np.array(val_y)), 
            ((np.array(test_x_1), np.array(test_x_2)), np.array(test_y))
        )
    else:
        return (
            ((np.array(train_x_1), np.array(train_x_2)), np.array(train_y)), 
            ((np.array(test_x_1), np.array(test_x_2)), np.array(test_y))
            )

def data_loader_v1(feature_type, val=True, scale=False, base_dir='EmotiW2023 Data Small'):
    
    """Data load without having separate npy files for splits
    """
    labels = pd.read_excel(f'{base_dir}/engagement_labels.xlsx', index_col=0) # Load label file
    print(labels.head())
    
    # OpenFace
    #Xy = np.load(f'{base_dir}/Xy_{feature_type}_10.npy', allow_pickle=True) # Load npy file

    # Marlin
    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True) # Load npy file
    
    Xy = utils.cleanXy(Xy)
    print(type(Xy), len(Xy))

    features_label_map = {}
    for xy in Xy:  
        features_label_map[xy[0]] = (xy[1], xy[2])
        
    train_x = []
    train_y = []
    if val:
        val_x = []
        val_y = []
        
    test_x = []
    test_y = []
    
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')
    print('TrainXy:', trainXy[:5])
    print('len TrainXy:', len(trainXy))
    print('TestXy:', testXy[:5])
    print('len TestXy:', len(testXy))
    print('ValXy:', valXy[:5])
    print('len ValXy:', len(valXy))

    # Debugging
    print("trainXy sample:", trainXy[:5])
    print("features_label_map keys sample:", list(features_label_map.keys())[:5])

    # Initialize error counters
    train_not_found_count = 0
    val_not_found_count = 0
    test_not_found_count = 0

    # Training loop
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
                train_x.append(xy[0])
                train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            print ('not found(train): ', k)
            train_not_found_count += 1
            # pass             
    if scale:    
        X = np.array(train_x)
        scaler.fit(X.reshape(-1, X.shape[-1]))
        for i in range(len(train_x)):
            train_x[i] = scaler.transform(train_x[i]) 

    # Validation loop
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
                     
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                if val:
                    val_x.append(x)
                    val_y.append(config.LABEL_MAP[xy[1]])
                else:
                    train_x.append(x)
                    train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            #pass
            print ('not found(val): ', k)
            val_not_found_count += 1

    # Test loop
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:                
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                test_x.append(x)
                test_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            #pass
            print ('not found(test): ', k)
            test_not_found_count += 1

    # Print total not found counts
    print(f"Total not found in train: {train_not_found_count}")
    print(f"Total not found in validation: {val_not_found_count}")
    print(f"Total not found in test: {test_not_found_count}")

    if val:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(val_x), np.array(val_y)), 
                (np.array(test_x), np.array(test_y)))
    else:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(test_x), np.array(test_y)))

if __name__ == '__main__':

    print ("testing data prep")
    feature_type = config.GAZE_HP_AU
    train, val, test = data_loader_v1(feature_type, val=True)
    train_x, train_y = train
    test_x, test_y = test
    val_x, val_y = val
    print(len(train_x), len(train_y), len(test_x), len(test_y), len(val_x), len(val_y))