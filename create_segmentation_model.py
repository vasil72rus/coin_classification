import numpy as np

from tensorflow.keras.layers import Lambda, Dense, Input, Flatten, MaxPool2D, Reshape, UpSampling2D, Conv2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, concatenate, Reshape
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# from vgg16 import VGG16
from tensorflow.keras.applications import VGG16, VGG19, NASNetLarge, InceptionV3, EfficientNetB7


def create_model():
  vgg_model = VGG19()

  inp = vgg_model.layers[-4].output
  x = Dense(4096, activation='relu')(inp)
  # x = Dense(8192, activation='relu')(x)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.25)(x)
  x = Dense(4096, activation='relu')(x)
  # x = Dense(8192, activation='relu')(x)
  # x = Dense(16384, activation='relu')(x)
  x = Dense(25088, activation='relu')(x)
  x = Reshape((7,7,512))(x)
  x = concatenate([x, vgg_model.layers[-5].output])
  x = Dropout(0.25)(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  x = concatenate([x, vgg_model.layers[-9].output])
  x = Dropout(0.25)(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Dropout(0.25)(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  x = concatenate([x, vgg_model.layers[-14].output])
  x = Dropout(0.25)(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Dropout(0.25)(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = Conv2D(512, 5, padding='same', activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  x = concatenate([x, vgg_model.layers[-19].output])
  x = Dropout(0.25)(x)
  x = Conv2D(256, 5, padding='same', activation='relu')(x)
  x = Conv2D(256, 5, padding='same', activation='relu')(x)
  x = Dropout(0.25)(x)
  x = Conv2D(256, 5, padding='same', activation='relu')(x)
  x = Conv2D(256, 5, padding='same', activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  x = concatenate([x, vgg_model.layers[-22].output])
  x = Dropout(0.25)(x)
  x = Conv2D(128, 5, padding='same', activation='relu')(x)
  x = Conv2D(128, 5, padding='same', activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  x = Dropout(0.25)(x)
  x = Conv2D(64, 5, padding='same', activation='relu')(x)
  x = Conv2D(64, 5, padding='same', activation='relu')(x)
  out = Conv2D(1, 5, padding='same', activation='sigmoid')(x)

  # decoder = Model(inp, out)

  # out =Dense(25088, activation='relu')(vgg_model.layers[-4].output)
  # encoder = Model(inputs=vgg_model.layers[0].input, outputs=vgg_model.layers[-4].output)

  # model = Model(vgg_model.layers[0].input, decoder(encoder(vgg_model.layers[0].input)))
  # vgg_model.trainable = False
  model = Model(vgg_model.layers[0].input, out)
  return model