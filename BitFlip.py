#!/usr/bin/env python
# coding: utf-8

# In[180]:


import random
import numpy as np


# In[181]:


temp = random.randrange(0, 784)
temp


# In[182]:


binary = int(f"{temp:b}")


# In[183]:


print (binary)
print (type(binary))

length = len(str(binary))
print (length)


# In[184]:


binary_str = f"{temp:b}"
print(type(binary_str))


# In[185]:


bitflip = random.randrange(1, length)
print(type(bitflip))
print('index number of Bit to be flipped = ', bitflip)

bit_to_be_flipped = int(bitflip)
print('Bit before flip = ', binary_str[bit_to_be_flipped])
print('Binary number before flip = ', binary)


# In[186]:


new = 0
count = 0
for i in binary_str:
    print(i)
    if(count == bitflip):
        if (int(i) > 0):
            new = 10*new + int(i)*0
        else:
            new = 10*new + int(i)+1 
        
    else:
        new = 10*new + int(i)*1 
    count = count+1
    
print('Binary number after flip = ', new)


# In[188]:


decimal = int(f"{new:d}",2)
print(decimal)


# In[ ]:




