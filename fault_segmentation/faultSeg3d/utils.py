import numpy as np
from tensorflow import keras
import random

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(96,96,96), 
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def _reshape_and_center_crop_or_pad(self, arr):
    raw_n = int(round(arr.size ** (1.0 / 3.0)))
    if raw_n ** 3 != arr.size:
      raise ValueError("Input .dat size is not a perfect cube: {}".format(arr.size))

    vol = np.reshape(arr, (raw_n, raw_n, raw_n))
    t1, t2, t3 = self.dim

    # Center crop when raw volume is larger than target.
    if raw_n >= t1 and raw_n >= t2 and raw_n >= t3:
      s1 = (raw_n - t1) // 2
      s2 = (raw_n - t2) // 2
      s3 = (raw_n - t3) // 2
      return vol[s1:s1+t1, s2:s2+t2, s3:s3+t3]

    # Fallback: center pad if target is larger than raw volume.
    out = np.zeros(self.dim, dtype=np.single)
    b1 = (t1 - raw_n) // 2
    b2 = (t2 - raw_n) // 2
    b3 = (t3 - raw_n) // 2
    out[b1:b1+raw_n, b2:b2+raw_n, b3:b3+raw_n] = vol
    return out

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    current_batch_size = len(data_IDs_temp)
    X = np.zeros((current_batch_size, *self.dim, self.n_channels), dtype=np.single)
    Y = np.zeros((current_batch_size, *self.dim, self.n_channels), dtype=np.single)

    for i, data_id in enumerate(data_IDs_temp):
      gx = np.fromfile(self.dpath + str(data_id) + '.dat', dtype=np.single)
      fx = np.fromfile(self.fpath + str(data_id) + '.dat', dtype=np.single)
      
      # convert fault indices to mask
      fx = (fx > 0).astype(np.single)
      
      gx = self._reshape_and_center_crop_or_pad(gx)
      fx = self._reshape_and_center_crop_or_pad(fx)

      # ==========================================
      # --- DATA AUGMENTATION ---
      # ==========================================
      
      # 1. Random Flipping (Inline / Crossline)
      # 50% chance to flip along the X-axis
      if random.random() > 0.5:
          gx = np.flip(gx, axis=0)
          fx = np.flip(fx, axis=0)
          
      # 50% chance to flip along the Y-axis
      if random.random() > 0.5:
          gx = np.flip(gx, axis=1)
          fx = np.flip(fx, axis=1)

      # 2. Random Gaussian Noise
      # 50% chance to inject noise into the seismic volume
      if random.random() > 0.5:
          # Scale noise severity between 5% and 15% of the patch's natural standard deviation
          noise_severity = random.uniform(0.05, 0.15)
          noise_std = np.std(gx) * noise_severity
          
          # Generate noise and add it strictly to the seismic image (gx), NOT the labels (fx)
          noise = np.random.normal(loc=0.0, scale=noise_std, size=gx.shape).astype(np.single)
          gx = gx + noise

      # Normalization
      xm = np.mean(gx)
      xs = np.std(gx)
      gx = gx - xm
      gx = gx / (xs + 1e-8)
      
      # Transpose to (z, y, x)
      gx = np.transpose(gx)
      fx = np.transpose(fx)

      X[i,] = np.reshape(gx, (*self.dim, self.n_channels))
      Y[i,] = np.reshape(fx, (*self.dim, self.n_channels))

    return X, Y