import os
import json
import time
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from utils.preprocessing import loadImg4Classification
from utils.model import PXRCmodel

with open('classification_info.json') as info:
    train_info = json.load(info)
    
    
batch_size           = batch = train_info['batch_size']
image_channel        = nc    = train_info['image_channel']
image_size           = isize = train_info['image_size']
epochs               = train_info['epochs']
gpu_number           = train_info['gpu_number']
adamLr               = train_info['adamLr']
test_dir             = train_info['test_dir']
train_dir            = train_info['train_dir']
valid_dir            = train_info['valid_dir']
transfer_weight_path = train_info['transfer_weight_path']
checkpoint_path      = train_info['checkpoint_path']
csv_logger_path      = train_info['csv_logger_path']
conv_init = RandomNormal(0, 0.02)




dir_name = '%s-%02d-%02d_%02d-%02d_train'%(time.localtime()[0], time.localtime()[1], time.localtime()[2], 
                                           time.localtime()[3], time.localtime()[4])

weight_dir = checkpoint_path+'PXRModelWeight/'+dir_name
log_dir    = csv_logger_path+'PXRModelLog/'+dir_name

try:
    os.mkdir(checkpoint_path+'PXRModelWeight')
    os.mkdir(csv_logger_path+'PXRModelLog')
except:
    pass

try:
    os.mkdir(weight_dir)
    os.mkdir(log_dir)
except:
    pass




os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)




# load the transfer learning model
tflr_model = load_model(transfer_weight_path)


# Load the training images
train = loadImg4Classification.train(isize, batch, train_dir)
valid = loadImg4Classification.valid(isize, batch, valid_dir)


# create the classification model
model = PXRCmodel(isize, nc, conv_init)
model.summary()


# transfer the weights
for i in range(len(tflr_model.layers)-4):
    model.layers[i+1].set_weights(tflr_model.layers[i+1].get_weights())
    

# compile the model
adam = Adam(lr=adamLr)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])


# Define csv logger to save the accuracies, losses ... of each epochs  
csv_logger = CSVLogger(log_dir+'/log.csv', append=True, separator=',')


# define checkpoint to save the weights while training. Some parameters can
# help to fine tune the weights
checkpoint = ModelCheckpoint(weight_dir+'/weight.h5', monitor='val_acc', verbose=1,
                             save_best_only = False, mode='max')


# training the model, and put the result into "history"
history = model.fit_generator(
    generator=train,
    epochs = epochs,
    steps_per_epoch = len(train),
    validation_data = valid,
    validation_steps = len(valid),
    callbacks = [csv_logger, checkpoint],
    verbose=1)

model.save(os.path.join(checkpoint_path))