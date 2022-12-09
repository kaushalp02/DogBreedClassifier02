#!/usr/bin/env python
# coding: utf-8

# In[26]:


# In[1]:


import gc
gc.collect()
import pandas as pd
import numpy as np
import os
fpath = r"C:\Users\himal\OneDrive\Desktop\mysite\Data\Images"
print(os.listdir(fpath))
dog_classes = os.listdir(fpath)


# In[2]:


breeds = [breed.split('-',1)[1] for breed in dog_classes]
breeds[:10]


# In[41]:


from itertools import chain
X = []
y = []
fullpaths = [r'C:\Users\himal\OneDrive\Desktop\mysite\Data\Images\{}'.format(dog_class) for dog_class in dog_classes]

for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        X.append([fullpath + '\\' + imgname])
        y.append(breeds[counter])
print(X[:10],"\n")
print(y[:10],"\n")

X =list(chain.from_iterable(X))
print(X[:10],"\n")
len(X)


# In[42]:


import random 
combined = list(zip(X,y))
print(combined[:10],"\n")
random.shuffle(combined)
print(combined[:10],"\n")
X[:], y[:] = zip(*combined)


# In[51]:


import matplotlib.pyplot as plt
from matplotlib.image import imread
plt.figure(figsize=(18,18))
for counter, i in enumerate(random.sample(range(0,len(X)),9)):
#     plt.subplot(3,3,counter+1)
#     plt.subplots_adjust(hspace=0.3)
    filename = X[i]
    image = imread(filename)
#     plt.imshow(image)
#     plt.title(y[i],fontsize=12)


# In[52]:


X = X[:1000]
y = y[:1000]


# In[ ]:





# In[53]:


import tensorflow
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
le.fit(y)
y_ohe = to_categorical(le.transform(y), len(breeds))
print(y_ohe.shape)
y_ohe = np.array(y_ohe)


# In[54]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
img_data = np.array([img_to_array(load_img(img, target_size = (299,299))) for img in X])
print(img_data.shape)

x_train, x_test, y_train, y_test = train_test_split(img_data, y_ohe,test_size=0.2,random_state= 2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=2)
import gc 
del img_data
gc.collect()


# In[55]:


from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  rotation_range = 30,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip= True)

train_generator = train_datagen.flow(x_train,y_train,shuffle=False,batch_size=batch_size,seed=1)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_datagerator = val_datagen.flow(x_val,y_val,shuffle=False,batch_size=batch_size,seed=1)


# In[56]:


img_id = 2
dog_generator = train_datagen.flow(x_train[img_id:img_id+1],y_train[img_id:img_id+1],
                                  shuffle=False,batch_size=batch_size,seed=1)
plt.figure(figsize=(20,20))
dogs = [next(dog_generator) for i in range(0,5)]
for counter, dog in enumerate(dogs):
    plt.subplot(1, 5,counter+1)


# In[57]:


from keras import models
from keras import layers 
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model

base_model = InceptionV3(weights = 'imagenet',include_top=False,input_shape=(299,299,3))

model = models.Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(len(breeds),activation='softmax'))
model.layers[0].trainable = False


# 

# In[58]:


model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:





# In[59]:


import pickle
model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
img = 'C:\\Users\\himal\\OneDrive\\Desktop\\mysite\\temp.png'

#     model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
img_data = np.array([img_to_array(load_img(img, target_size = (299,299,3)))])
x_test1 = img_data / 255.
test_predictions = model.predict(x_test1)
print(test_predictions)
predictions = le.classes_[np.argmax(test_predictions,axis=1)]
print(predictions)


# train_steps_per_epoch = x_train.shape[0]//batch_size
# val_steps_per_epoch = x_val.shape[0]//batch_size
# epochs=20
# history = model.fit_generator(train_generator,
#                              steps_per_epoch = train_steps_per_epoch,
#                              validation_steps = val_steps_per_epoch,
#                              epochs= epochs,verbose=1)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




