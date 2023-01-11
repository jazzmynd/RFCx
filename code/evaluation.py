#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping

# from datagen import get_files_and_labels, scalespec, preprocess, DataGenerator
from datagen_test import TestDataGenerator
from datagen import get_files_and_labels, DataGenerator
from learningrate import warmup_cosine_decay, WarmUpCosineDecayScheduler
from specinput import load_audio, wave_to_mel_spec
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import model_from_json
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

  
mode =  (sys.argv[1])
model_path = sys.argv[2]
if model_path[-1] == '/':
    model_path = model_path[0:-1]
data_dir = sys.argv[3]
if data_dir[-1]== '/':
    data_dir = data_dir[0:-1]
                       

# load model
# model_architecture_path = './baseline_model/model.json'
# model_weight_path = './baseline_model/model_best_val.h5' # path of model
model_architecture_path = './'+model_path+'/model.json'
model_weight_path = './'+model_path+'/model_best_val.h5' # path of model
# model_weight_path = './'+model_path+'/model0020.h5' # path of model

model = model_from_json(open(model_architecture_path).read()) # load architecture
model.load_weights(model_weight_path) # load weights

# specify list of target classes
# data_dir = "../image_Data/puerto-rico/train/audio"
# class_list = os.listdir(data_dir+'/p/')
class_list = ['Coereba_flaveola', 'Eleutherodactylus_coqui', 'Spindalis_portoricensis', 'Contopus_latirostris_blancoi',
              'Eleutherodactylus_antillensis', 'Amazona_vittata', 'Setophaga_discolor', 'Margarops_fuscatus', 'Melopyrrha_portoricensis',
              'Eleutherodactylus_brittoni', 'Geotrygon_montana', 'Myiarchus_antillarum', 'Pluvialis_squatarola', 'Patagioenas_leucocephala',
              'Buteo_platypterus', 'Coccyzus_vieilloti', 'Lithobates_catesbeianus', 'Setophaga_petechia', 'Icterus_icterus',
              'Crotophaga_ani', 'Antrostomus_noctitherus', 'Eleutherodactylus_cochranae', 'Eleutherodactylus_wightmanae',
              'Setophaga_adelaidae', 'Patagioenas_squamosa', 'Nesospingus_speculiferus', 'Melanerpes_portoricensis',
              'Leptodactylus_albilabris', 'Rhinella_marina', 'Megascops_nudipes', 'Setophaga_angelae', 'Eleutherodactylus_richmondi',
              'Eleutherodactylus_hedricki', 'Eleutherodactylus_gryllus', 'Osteopilus_septentrionalis', 'Turdus_plumbeus',
              'Vireo_latimeri', 'Eleutherodactylus_unicolor', 'Chordeiles_gundlachii', 'Vireo_altiloquus', 'Eleutherodactylus_cooki',
              'Eleutherodactylus_portoricensis', 'Molothrus_bonariensis', 'Todus_mexicanus', 'Buteo_jamaicensis']

if mode.lower() == "test":
    # load test data
    # test_data_path = '../image_data/puerto-rico/test/audio/'
    # test_labels=pd.read_csv('test-labels.csv')
    # test_labels=test_labels.rename(columns={"Unnamed: 0": "id"})
    # test_data_path = '../image_Data/puerto-rico/test/audio/'
    files_test=[]
    for test_id in os.listdir(data_dir):
        test_data_path=data_dir+'/'+test_id
        for f in os.listdir(test_data_path):
            files_test.append(test_data_path+'/'+f)


    resize_dim = [224, 224] # desired shape of generated images
    batch_size = 2 # len(files_test)
    augment = 0  # whether to apply data augmentation
    # no aug on test data
    # test data generator
    test_generator = TestDataGenerator(files_test,
                                       resize_dim=resize_dim,
                                       batch_size=batch_size,
                                       augment=augment,
                                       shuffle=False)
#     pred_prob = model.predict(test_generator[0][0])
#     pred_label = 1*(pred_prob>0.5)
#     files = [path.split('/')[-1] for path in test_generator[0][1]]
#     pred_prob_all = (np.column_stack((files, pred_label)))
#     pred_label_all = (np.column_stack((files, pred_label)))
    files=[]
    for batch in range(len(test_generator)):
        pred_prob = model.predict(test_generator[batch][0])
        pred_label = 1*(pred_prob>0.5)
        files = [path.split('/')[-1][0:-4] for path in test_generator[batch][1]]
#         labels_expend = np.column_stack((np.zeros(batch_size),test_generator[batch][1]))
#         real_labels = labels_expend.argmax(axis=1)-1 # -1 means negative sample
        if batch > 0:
            pred_label_all = np.concatenate((pred_label_all, np.column_stack((files, pred_label))), axis=0)
            pred_prob_all = np.concatenate((pred_prob_all, np.column_stack((files, pred_prob))), axis=0) 
        else:
            pred_label_all = np.column_stack((files, pred_label))
            pred_prob_all = np.column_stack((files, pred_prob))
    
    pred_label_df = pd.DataFrame(pred_label_all, columns =['id']+class_list)
    pred_label_df.to_csv(model_path+"/prediction_label_test.csv", index=False)
    pred_prob_df = pd.DataFrame(pred_prob_all, columns =['id']+class_list)
    pred_prob_df.to_csv(model_path+"/prediction_prob_test.csv", index=False)
    
    
if mode.lower() == "val":
    # load train data
    train_split = 1.0

    # generate positive train file paths
    files_train_p, files_val_p, labels = get_files_and_labels(data_dir+'/val/p/',
                                                          train_split=train_split,
                                                          random_state=42,
                                                          classes=class_list)

    # generate negative train file paths
    files_train_n, files_val_n, labels_n = get_files_and_labels(data_dir+'/val/n/',
                                                            train_split=train_split,
                                                            random_state=42,
                                                            classes=class_list) 

    labels_rev = dict((v,k) for (k,v) in labels.items())
    files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
    files_train = files_train_p+files_train_n


    resize_dim = [224, 224] # desired shape of generated images
    augment = 0 # whether to apply data augmentation
    batch_size = 2 # len(files_train)
    # train data generator
    train_generator = DataGenerator(files_train, labels,
                                resize_dim=resize_dim,
                                batch_size=batch_size,
                                augment=augment)
#     print(len(train_generator))
    for batch in range(len(train_generator)):
        pred_prob = model.predict(train_generator[batch][0])
        pred_label = 1*(pred_prob>0.5)
        labels_expend = np.column_stack((np.zeros(batch_size),train_generator[batch][1]))
        real_labels = labels_expend.argmax(axis=1)-1 # -1 means negative sample
        if batch > 0:
            pred_label_all = np.concatenate((pred_label_all, np.column_stack((real_labels, pred_label))), axis=0)
            pred_prob_all = np.concatenate((pred_prob_all, np.column_stack((real_labels, pred_prob))), axis=0) 
        else:
            pred_label_all = np.column_stack((real_labels, pred_label))
            pred_prob_all = np.column_stack((real_labels, pred_prob))
    
    pred_label_df = pd.DataFrame(pred_label_all, columns =['Real Label']+class_list)
    pred_label_df.to_csv(model_path+"/prediction_label_val.csv", index=False)
    
    pred_prob_df = pd.DataFrame(pred_prob_all, columns =['Real Label']+class_list)
    pred_prob_df.to_csv(model_path+"/prediction_prob_val.csv", index=False)
    

if mode.lower() == "train":
    # load train data
    train_split = 1.0

    # generate positive train file paths
    files_train_p, files_val_p, labels = get_files_and_labels(data_dir+'/train/p/',
                                                          train_split=train_split,
                                                          random_state=42,
                                                          classes=class_list)

    # generate negative train file paths
    files_train_n, files_val_n, labels_n = get_files_and_labels(data_dir+'/train/n/',
                                                            train_split=train_split,
                                                            random_state=42,
                                                            classes=class_list) 

    labels_rev = dict((v,k) for (k,v) in labels.items())
    files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
    files_train = files_train_p+files_train_n


    resize_dim = [224, 224] # desired shape of generated images
    augment = 0 # whether to apply data augmentation
    batch_size = 2 # len(files_train)
    # train data generator
    train_generator = DataGenerator(files_train, labels,
                                resize_dim=resize_dim,
                                batch_size=batch_size,
                                augment=augment)
#     print(len(train_generator))
    for batch in range(len(train_generator)):
        pred_prob = model.predict(train_generator[batch][0])
        pred_label = 1*(pred_prob>0.5)
        labels_expend = np.column_stack((np.zeros(batch_size),train_generator[batch][1]))
        real_labels = labels_expend.argmax(axis=1)-1 # -1 means negative sample
        if batch > 0:
            pred_label_all = np.concatenate((pred_label_all, np.column_stack((real_labels, pred_label))), axis=0)
            pred_prob_all = np.concatenate((pred_prob_all, np.column_stack((real_labels, pred_prob))), axis=0) 
        else:
            pred_label_all = np.column_stack((real_labels, pred_label))
            pred_prob_all = np.column_stack((real_labels, pred_prob))
    
    pred_label_df = pd.DataFrame(pred_label_all, columns =['Real Label']+class_list)
    pred_label_df.to_csv(model_path+"/prediction_label_train.csv", index=False)
    
    pred_prob_df = pd.DataFrame(pred_prob_all, columns =['Real Label']+class_list)
    pred_prob_df.to_csv(model_path+"/prediction_prob_train.csv", index=False)
        
        
        
 

