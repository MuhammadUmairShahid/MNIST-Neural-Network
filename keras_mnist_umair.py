# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 00:30:32 2020

@author: syed
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:45:18 2020

@author: syed
"""
# In[1]:

import numpy as np
import matplotlib.pyplot as plt 
import time
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import normalize
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import six
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# step 3 Normalise the data 
# we can either use the following function to normalise, or we can just divide it by the highest value
X_train = X_train/255.0
X_test = X_test/255.0

first_image = X_train[0]
first_image.shape
plt.imshow(first_image, cmap = 'gray')


# In[keras base model ]:

# Now all the fun Tensorflow stuff
# Build the model

i = Input(shape= X_train[0].shape)
x = Flatten()(i)

x = Dense(32, name='dense_1')(x)
x = Activation("relu", name="act_1")(x)

x = Dense(32, name='dense_2')(x)
x = Activation("relu", name="act_2")(x)

#x = Dense(16, name='dense_3')(x)
#x = Activation("relu", name="act_3")(x)

x = Dense(10, name='output_layer')(x)
x = Activation("softmax", name="softmax")(x)

model = Model(i, x)
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


r= model.fit(X_train, y_train,  validation_split= 0.20, epochs=50)


# Plot what's returned by model.fit()
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.summary()
