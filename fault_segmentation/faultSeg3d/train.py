import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Must be set before importing tensorflow.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

np.random.seed(12345)
random.seed(12345)
tf.random.set_seed(1234)

def configure_gpu():
  gpus = tf.config.list_physical_devices('GPU')
  if not gpus:
    return
  try:
    # Keep training on one GPU to avoid multi-device allocator pressure.
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError:
    # TF raises if device config is called after runtime initialization.
    pass

configure_gpu()
from utils import DataGenerator
from unet3 import *

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,
          'dim':(96,96,96),
          'n_channels':1,
          'shuffle': True}

# download the data and put it in the data folder
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"

  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  train_ID = list(range(200))
  train_ID = train_ID * 4
  valid_ID = range(20)
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  tf.keras.backend.clear_session()
  model = unet(input_size=(None, None, None,1))
  model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', 
                metrics=['accuracy'])
  model.summary()

  # checkpoint
  filepath="check1/fseg-{epoch:02d}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
        verbose=1, save_best_only=False, mode='max')
  logging = TensorBoard(log_dir='./log1')
  #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
  #                              patience=20, min_lr=1e-8)
  callbacks_list = [checkpoint, logging]
  print("data prepared, ready to train!")
  # Fit the model
  history=model.fit(x=train_generator,
  validation_data=valid_generator,epochs=200,callbacks=callbacks_list,verbose=2,
  max_queue_size=1, workers=1, use_multiprocessing=False)
  model.save('check1/fseg.hdf5')
  # showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

  # summarize history for loss
  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

if __name__ == '__main__':
    main()

