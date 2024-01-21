#!/usr/bin/env python
# coding: utf-8

# ## Twin Classification
# 
# ### 1. Install Packages and Libraries

# In[1]:


import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


# In[2]:


# To import and transform images 
from keras.preprocessing import image_dataset_from_directory

# To build the NN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# To save the model
from tensorflow.keras.models import load_model


# ## 2. Access Image data in data folder

# In[3]:


batchsize=2
data_directory = "data"
data = image_dataset_from_directory(data_directory, batch_size=batchsize)


# In[4]:


data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Kisha images are classified as 0 and Kym images are classified as 1
# 
# ## 3. Scale and Split the Data

# In[6]:


# Scale the data
data = data.map(lambda x,y: (x/255,y))

#Split Data
train_size = int(len(data)*.6)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.2)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# ## 4. Build the Deep Learning Model

# In[7]:


# Input Layer, 2 Convolutional Hidden Layers, a Flatten and Dense Layer and Output Layer 
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[8]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[9]:


# ## 5. Train The model

# In[10]:


# for Logging
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# fitting the model with the training data
hist = model.fit(train, epochs=10, validation_data=val, callbacks =[tensorboard_callback])


# ## 6. Model Analysis

#
# ## 7. Save and Register the Model

# In[13]:


# Save the model
model.save(os.path.join('models', 'twinclassifier.h5'))
#model.save('twinclassifier.keras')


# In[14]:


# Register the model to the workspace
#model = run.regster_model(model_name='twin_classifier', model_path='./models/twinclassifier.h5')
#print(model.name, model.id, model.version, sep='\t')


# ## Create a requirements file based on notebook dependencies

# In[14]:
