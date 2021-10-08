import pickle
import numpy as np
import pandas as pd

# import efficientnet.keras as efn

# from __future__ import division
# from __future__ import print_function

from tensorflow.keras.layers import Lambda, Dense, TimeDistributed, Input, Flatten, MaxPool2D, Reshape, UpSampling2D, Conv2D
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
from tensorflow.keras.applications.vgg19 import preprocess_input

import scipy.io
from scipy.spatial.distance import braycurtis, cosine, canberra, chebyshev, cityblock, correlation, euclidean, mahalanobis, minkowski, seuclidean, sqeuclidean, wminkowski 
import os
import matplotlib.pyplot as plt
import cv2
from time import time

def contrast(img, contrast):
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(5,5))
    return clahe.apply(img)

def get_edges(img):
    img = np.uint8(img)
    img1 = contrast(img, 1)
    img2 = contrast(img, 2)
    img3 = contrast(img, 3)
    return np.stack([img1, img2, img3], axis=2)


def load_image(path):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (224, 224))

    return img
	
df = pd.read_csv('csv/catalog.csv', encoding='utf-8')

vgg_model = VGG19(weights='imagenet')
model = Model(vgg_model.input, vgg_model.layers[-4].output)

images = df.values[:,0]

def train_predict(i, batch_size):
    batch = []
    for j in range(i, i+batch_size):
        try:
            batch.append(get_edges(load_image(f'images/{images[j]}_a.jpg')))
        except:
            batch.append(get_edges(load_image(f'images/{images[0]}_a.jpg')))
    batch = np.asarray(batch)
    # augmen_images = np.array(list(map(get_edges, batch)))
    # augmen_images = np.array(list(map(preprocess_input, augmen_images)))
    predict_a = model.predict(batch)
    batch = []
    for j in range(i, i+batch_size):
        try:
            batch.append(get_edges(load_image(f'images/{images[j]}_r.jpg')))
        except:
            batch.append(get_edges(load_image(f'images/{images[0]}_r.jpg')))
    batch = np.asarray(batch)
    # augmen_images = np.array(list(map(get_edges, batch)))
    # augmen_images = np.array(list(map(preprocess_input, augmen_images)))
    predict_r = model.predict(batch)
    return (predict_a * predict_r)
	
batch_size = 8
from time import time
t = time()
embeddings = train_predict(0, batch_size)
print('start compute embedddings')
for i in range(batch_size, 30000, batch_size):
    embedding= train_predict(i, batch_size)
    embeddings = np.concatenate((embeddings, embedding))
    print(embeddings.shape)
    if i % 500 == 0:
        print(embeddings.shape)
        print('Успешно!', i)

np.save('embeddings_CCC_VGG19.npy', embeddings)
print('All time:', time()-t)
