#!/usr/bin/env python
# coding: utf-8


import os
import json
import numpy as np
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping

from datagen import get_files_and_labels, scalespec, preprocess, DataGenerator
from learningrate import warmup_cosine_decay, WarmUpCosineDecayScheduler
from specinput import load_audio, wave_to_mel_spec
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# In[4]:

data_dir = sys.argv[1]
if data_dir[-1]=="/":
    data_dir = data_dir[0:-1] 
    
model_out = sys.argv[2] 
if model_out[-1]!="/":
    model_out += "/"

if not os.path.exists(model_out):
    os.mkdir(model_out)
    
if len(sys.argv) == 4:
    epochs = int(sys.argv[3])
else:
    epochs = 100

print(data_dir, epochs)

model_out += "model"
# model_out = './model' # path to output model
# data_dir = "../image_Data/puerto-rico/train/audio"
# expected format:
#     data_dir/
#         p/
#             <class_1>/
#                 <audio_filename>.wav
#                 ...
#             <class_2>/
#                 <audio_filename>.wav
#                 ...
#         n/
#             <class_1>/
#                 <audio_filename>.wav
#                 ...
#             <class_2>/
#                 <audio_filename>.wav
#                 ...


# In[5]:


# specify list of target classes
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
num_classes = len(class_list)

# Get training file paths

# In[6]:


train_split = 0.8

# generate positive train file paths
files_train_p, files_val_p, labels = get_files_and_labels(data_dir+'/p/',
                                                          train_split=train_split,
                                                          random_state=42,
                                                          classes=class_list)

# generate negative train file paths
files_train_n, files_val_n, labels_n = get_files_and_labels(data_dir+'/n/',
                                                            train_split=train_split,
                                                            random_state=42,
                                                            classes=class_list) 

labels_rev = dict((v,k) for (k,v) in labels.items())
files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
files_train = files_train_p+files_train_n
files_val = files_val_p+files_val_n


# Setup data generator

# In[7]:


resize_dim = [224, 224] # desired shape of generated images
augment = 0 # whether to apply data augmentation
batch_size = 32

# train data generator
train_generator = DataGenerator(files_train,
                                labels,
                                resize_dim=resize_dim,
                                batch_size=batch_size,
                                augment=augment)

# validation data generator
val_generator = DataGenerator(files_val,
                              labels,
                              resize_dim=resize_dim,
                              batch_size=batch_size,
                              augment=0)


# Define model

# In[8]:


conv = MobileNetV2(weights=None, 
                   include_top=False, 
                   input_shape=(224, 224, 3))

for layer in conv.layers:
    layer.trainable = True

model = models.Sequential()
model.add(conv)
model.add(layers.AveragePooling2D((7, 7)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam()

# note: this loss can be used to avoid assumptions about unlabeled classes
# def masked_loss(y_true, y_pred):
#     return K.mean(K.mean(K.binary_crossentropy(tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true),
#                                                tf.multiply(y_pred, tf.cast(tf.logical_not(tf.math.is_nan(y_true)), tf.float32))), axis=-1))

model.compile(loss='binary_crossentropy', 
              optimizer=optimizer)

model.summary()


# In[9]:


# save model architecture
model_json = model.to_json()
with open(model_out+'.json', "w") as json_file:
    json_file.write(model_json)
with open(model_out+'_classes.json', 'w') as f:
    json.dump(labels_rev, f)
print('Saved model architecture')


# These parameters specify the shape of a learning rate curve that has warmup and cosine decay. See learningrate.py for more details.

# In[10]:

warmup_lr = 1e-5
warmup_epochs = int(epochs*0.1)
patience = epochs
steps_per_epoch = len(train_generator)
base_lr = 0.0015
hold_base_rate_steps = int(epochs*0.125*steps_per_epoch)

total_steps = int(epochs * steps_per_epoch)
warmup_steps = int(warmup_epochs * steps_per_epoch)


# Setup callbacks

# In[12]:


# save the best model weights based on validation loss
val_chkpt = ModelCheckpoint(filepath=model_out+'_best_val.h5',
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1)

# also save the model weights every 20 epochs
reg_chkpt = ModelCheckpoint(filepath=model_out+'{epoch:04d}.h5',
                            save_weights_only=True,
                            save_freq=int(steps_per_epoch*20))

# apply a learning rate schedule
cosine_warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base= base_lr,
                                               total_steps= total_steps,
                                               warmup_learning_rate= warmup_lr,
                                               warmup_steps= warmup_steps,
                                               hold_base_rate_steps=hold_base_rate_steps)

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-5,
    patience=10,
)

callbacks_list = [val_chkpt, reg_chkpt, cosine_warm_up_lr, early_stop]


# Train

# In[13]:


model_history = model.fit(train_generator,
                          steps_per_epoch = len(train_generator),
                          validation_data = val_generator,
                          epochs = epochs,
                          verbose = 1,
                          callbacks=callbacks_list)
np.save(model_out+'_history.npy', model_history.history)






