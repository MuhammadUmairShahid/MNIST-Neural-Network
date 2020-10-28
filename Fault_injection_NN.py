#!/usr/bin/env python
# coding: utf-8

# # LOADING mnist MODEL AND TRAINING neural network

# In[111]:


#importing the relevent libraries
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
import struct


# In[112]:


#loading the predefined mnist dataset from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[179]:


#loading my trained model
model=load_model('mnist_neural_network.h5')


# In[180]:


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


# ## Displaying the accuracy before injecting faults

# In[181]:


#accuracy of model on train set
test_loss, train_acc = model.evaluate(X_train,  y_train, verbose=2)

print('\nTrain accuracy:', train_acc)


# In[182]:


#accuracy of model on test set
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# # --------------------------------- Fault Injection Code --------------------------------

# In[183]:


#function to convert binary digit into float
def bin2float(b):
    h = int(b, 2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]


#function to convert float into 64 bit binary digit
def float2bin(f):
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'


# In[184]:


#loading the weights of neural network
temp1= model.get_weights()
temp1


# In[185]:


#randomly selecting an index from weights array accross multiple layers
first_index = random.randrange(2,6,2)
second_index = random.randrange(0, 2)
third_index = random.randrange(0, 31)
flt = temp1[first_index][second_index][third_index]
print('float= ' ,flt)
print(type(flt))


# In[186]:


#converting the acquired float into binary equivalent
binary = float2bin(flt)

print (binary)
print (type(binary))

length = len(binary)
print (length)


# ## Applying the bitflip randomly

# In[187]:


#randomly selecting the bit to be flipped
bitflip = random.randrange(0, 32)
print(type(bitflip))
print('index number of Bit to be flipped  = ', bitflip)

print('Bit before flip = ', binary[bitflip])
print('Binary number before flip = ', binary)


# In[188]:


#code to flip the decided bit from binary number 
new = 0
count = 0
for i in binary:
    
    if(count == bitflip):
        if (int(i) > 0):
            new = 10*new + int(i)*0
        else:
            new = 10*new + int(i)+1 
        
    else:
        new = 10*new + int(i)*1 
    count = count+1
    
print('Binary number after flip = ', new)


# In[189]:


#new float received after bitflip
faulty_num = bin2float(str(new))
print(faulty_num)


# In[190]:


#inserting the faulty weight into weights array
temp1[first_index][second_index][third_index]= faulty_num


# In[191]:


#updating the weights into model
model.set_weights(temp1)

temp2= model.get_weights()

#weight before bitflip
print('Weight before bitflip', flt)

#weight after bitflip
print('Weight after bitflip', temp2[first_index][second_index][third_index])


# ## Using a for loop to randomly inject fault in multiple layers accross the model

# In[192]:


for i in range(2000):
    temp1= model.get_weights()
    
    first_index = random.randrange(2,4,2)
    second_index = random.randrange(0, 2)
    third_index = random.randrange(0, 31)
    flt = temp1[first_index][second_index][third_index]
    
    binary = float2bin(flt)
    bitflip = random.randrange(0, 32)
    
    new = 0
    count = 0
    for i in binary:
    
        if(count == bitflip):
            if (int(i) > 0):
                new = 10*new + int(i)*0
            else:
                new = 10*new + int(i)+1 
        
        else:
            new = 10*new + int(i)*1 
        count = count+1
    
    faulty_num = bin2float(str(new))
    temp1[first_index][second_index][third_index]= faulty_num
    
    model.set_weights(temp1)


# In[193]:


#Plot what's returned by model

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.summary()


# In[194]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# ## Increasing faults furthermore

# In[195]:


for i in range(8000):
    temp1= model.get_weights()
    
    first_index = random.randrange(2,4,2)
    second_index = random.randrange(0, 2)
    third_index = random.randrange(0, 31)
    flt = temp1[first_index][second_index][third_index]
    
    binary = float2bin(flt)

    bitflip = random.randrange(0, 32)
    
    new = 0
    count = 0
    for i in binary:
    
        if(count == bitflip):
            if (int(i) > 0):
                new = 10*new + int(i)*0
            else:
                new = 10*new + int(i)+1 
        
        else:
            new = 10*new + int(i)*1 
        count = count+1
    
    faulty_num = bin2float(str(new))
    temp1[first_index][second_index][third_index]= faulty_num
    
    model.set_weights(temp1)


# In[196]:


#Plot what's returned by model

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.summary()


# In[197]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# ## Testing the limits by injecting large amount of faults

# In[198]:


for i in range(200000):
    temp1= model.get_weights()
    
    first_index = random.randrange(2,4,2)
    second_index = random.randrange(0, 2)
    third_index = random.randrange(0, 31)
    flt = temp1[first_index][second_index][third_index]
    
    binary = float2bin(flt)
    #print(binary)
    bitflip = random.randrange(0, 32)
    
    new = 0
    count = 0
    for i in binary:
    
        if(count == bitflip):
            if (int(i) > 0):
                new = 10*new + int(i)*0
            else:
                new = 10*new + int(i)+1 
        
        else:
            new = 10*new + int(i)*1 
        count = count+1
    
    #print(new)
    faulty_num = bin2float(str(new))
    #print(faulty_num)
    temp1[first_index][second_index][third_index]= faulty_num
    
    model.set_weights(temp1)


# In[199]:


#Plot what's returned by model

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.summary()


# In[200]:


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[202]:


image_index = 5555
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
p = model.predict(X_test[image_index].reshape(1, 28, 28))
print('Model prediction = ', p.argmax())


# In[ ]:




