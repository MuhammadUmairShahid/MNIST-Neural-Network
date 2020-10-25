#!/usr/bin/env python
# coding: utf-8

# In[150]:


import numpy as np
import matplotlib.pyplot as plt 
import time
import random
import math
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


# In[218]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[219]:


model=load_model('mnist_neural_network.h5')


# In[153]:


image_index = 1000
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[154]:


image_index = 4567
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[155]:


image_index = 5555
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[156]:


image_index = 9732
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[157]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[158]:


test_loss, test_acc = model.evaluate(X_train,  y_train, verbose=2)

print('\nTest accuracy:', test_acc)


# In[159]:


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


# In[160]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[161]:


test_loss, test_acc = model.evaluate(X_train,  y_train, verbose=2)

print('\nTest accuracy:', test_acc)


# In[162]:


temp1= model.get_weights()
temp1


# In[184]:


first_index = random.randrange(2,6,2)
second_index = random.randrange(0, 2)
third_index = random.randrange(0, 31)
flt = temp1[first_index][second_index][third_index]
print('float= ' ,flt)
temp = round(flt)
print('Rounded off nearest integer= ' ,temp)


# In[185]:


if (temp > 0):
    temp1[first_index][second_index][third_index] = 0
    
else:
    temp1[first_index][second_index][third_index] = 1
    


# In[186]:


model.set_weights(temp1)

temp2= model.get_weights()
temp2[first_index][second_index][third_index]


# In[192]:


for i in range(1000):
    temp1= model.get_weights()
    
    first_index = 2
    second_index = random.randrange(0, 2)
    third_index = random.randrange(0, 31)
    flt = temp1[first_index][second_index][third_index]
    #print('float= ' ,flt)
    temp = round(flt)
    #print('Rounded off nearest integer= ' ,temp)
    
    if (temp > 0):
        temp1[first_index][second_index][third_index] = 0
    
    else:
        temp1[first_index][second_index][third_index] = 1
    
    model.set_weights(temp1)


# In[193]:


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


# In[194]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[224]:


for i in range(2000):
    temp1= model.get_weights()
    
    first_index = random.randrange(2,4,2)
    second_index = random.randrange(0, 2)
    third_index = random.randrange(0, 31)
    flt = temp1[first_index][second_index][third_index]
    #print('float= ' ,flt)
    temp = round(flt)
    #print('Rounded off nearest integer= ' ,temp)
    
    if (temp > 0):
        temp1[first_index][second_index][third_index] = 0
    
    else:
        temp1[first_index][second_index][third_index] = 1
    
    model.set_weights(temp1)


# In[225]:


r= model.fit(X_test, y_test,  validation_split= 0.20, epochs=30)


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


# In[226]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[223]:


image_index = 4567
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[ ]:




