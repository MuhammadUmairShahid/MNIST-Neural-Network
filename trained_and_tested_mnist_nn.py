#!/usr/bin/env python
# coding: utf-8

# In[32]:


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


# In[6]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[4]:


model=load_model('mnist_neural_network.h5')


# In[7]:


image_index = 1000
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[8]:


image_index = 4567
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[9]:


image_index = 5555
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[10]:


image_index = 9732
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[14]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[15]:


test_loss, test_acc = model.evaluate(X_train,  y_train, verbose=2)

print('\nTest accuracy:', test_acc)


# In[18]:


r= model.fit(X_test, y_test,  validation_split= 0.20, epochs=10)


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


# In[24]:


temp= model. get_weights()
temp


# In[39]:


a = tf.keras.layers.Dense(1,
  kernel_initializer=tf.constant_initializer(1.))
a_out = a(tf.convert_to_tensor([[0., 1., 0.]]))
a.get_weights()
[np.array([[1.],
       [1.],
       [1.]], dtype='f'), np.array([0.], dtype='f')]
b = tf.keras.layers.Dense(1,
  kernel_initializer=tf.constant_initializer(1.))
b_out = b(tf.convert_to_tensor([[0., 0., 0.]]))
b.get_weights()
[np.array([[2.],
       [2.],
       [2.]], dtype='f'), np.array([0.], dtype='f')]
b.set_weights(a.get_weights())
b.get_weights()
[np.array([[1.],
       [1.],
       [1.]], dtype='f'), np.array([0.], dtype='f')]


# In[37]:


r= model.fit(X_test, y_test,  validation_split= 0.20, epochs=10)


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


# In[38]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[40]:


image_index = 4567
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[ ]:




