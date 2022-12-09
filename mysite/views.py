

from importlib.metadata import requires
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.shortcuts import render
from keras import models
from keras import layers

from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pyrebase
import gc
gc.collect()
import pandas as pd
import numpy as np
import os
from itertools import chain
import random
import tensorflow
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
import requests
from PIL import Image
import shutil
import urllib
from django.contrib import auth
import numpy as np
import urllib.request
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img

user={}
city = ""

# go back to index main page from about 
def back(request):
    global user
    uid = user['localId']
    name = database.child("users").child(uid).child("details").child('name').get().val()
    session_id = user["idToken"]
    request.session["uid"]=str(session_id
                               )
    return render(request,'index.html',{"e":name})

# login process  
def index(request):
    email=request.POST.get("uname")
    passw = request.POST.get("psw") 
    
    try:
        global user
        user = authe.sign_in_with_email_and_password(email,passw)
        uid = user['localId']
        name = database.child("users").child(uid).child("details").child('name').get().val()
        global city
        city = database.child("users").child(uid).child("details").child('city').get().val()
    except:
        message = "Invalid Credentials"
        return render(request,'login.html',{"messg":message})

    session_id = user["idToken"]
    request.session["uid"]=str(session_id)
    return render(request,'index.html',{"e":name})

# sign up process 
def index_2(request):
    
    email = request.POST.get("uname")
    passw = request.POST.get("psw")
    city = request.POST.get("city")
    name = request.POST.get("name") 

    try:
        user = authe.create_user_with_email_and_password(email,passw)
        uid = user['localId']
        data={"Id":email,"name":name,"status":"1","city":city,"dp":"https://firebasestorage.googleapis.com/v0/b/mydogbreed-6ae22.appspot.com/o/profile.png?alt=media&token=1942831a-82fe-4c4c-b932-e2d7b7374e81"}
        database.child("users").child(uid).child("details").set(data)
        message = "Account Created Successfully"
    except:
        message_1 = "Account already exist or Invalid password"
        return render(request,'signup.html',{"messg":message_1})

    
    return render(request,'login.html',{"messg":message})    
    
# calling login page 
def login(request):
    return render(request,'login.html')

# logout button 
def loggedout(request):
    return render(request,'loggedout.html')

#about page 
def about(request):
    return render(request,'about.html')

# sign up  
@csrf_exempt
def signup(request):
    return render(request,'signup.html')

firebaseConfig = {
  "apiKey": "AIzaSyBOlfMCxj6qUnDrmQ-YivwzNEJoX4cVsWI",
  "authDomain": "mydogbreed-6ae22.firebaseapp.com",
  "projectId": "mydogbreed-6ae22",
  "storageBucket": "mydogbreed-6ae22.appspot.com",
  "messagingSenderId": "638367149171",
  "appId": "1:638367149171:web:83d5fa456fb4b6e46f8e06",
  "measurementId": "G-XJC8MCSZ3V",
  "databaseURL": "https://mydogbreed-6ae22-default-rtdb.asia-southeast1.firebasedatabase.app"
}
firebase = pyrebase.initialize_app(firebaseConfig)

authe = firebase.auth()
database = firebase.database()

def postsign(request):
    return render(request,'index.html')

# logout 
def logout(request):
    auth.logout(request)
    return render(request,'loggedout.html')

#community page 

def community(request):
    return render(request,'community.html')

#nearbt profile page
def nearby(request):
    global city
    global user
    uid = user['localId']
    all_user = list(database.child('users').shallow().get().val())
    i = 0
    near_user = []
    while i < len(all_user):
        if all_user[i] != uid:
            Local_city = database.child("users").child(all_user[i]).child("details").child('city').get().val()
        
            if Local_city == city:
                near_user.append(all_user[i])
                # Local_name = database.child("users").child(all_user[i]).child("details").child('name').get().val()
        i = i +1
    local_names = []
    local_emails = []
    dps = []
    for i in near_user:
        Local_name = database.child("users").child(i).child("details").child('name').get().val()
        local_names.append(Local_name)
        Local_email = database.child("users").child(i).child("details").child('Id').get().val()
        local_emails.append(Local_email)
        Local_dp = database.child("users").child(i).child("details").child('dp').get().val()
        dps.append(Local_dp)
    comb_lis = zip(local_emails,local_names,dps)

    return render(request,'nearby_profiles.html',{'comb_lis':comb_lis})


#my profile upload photos 
def myprofile(request):
    return render(request,'profile.html')

#for dp
def mydp(request):
    return render(request,'setdp.html')

def geturl(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    import random
    try:
        all_img = list(database.child('users').child(uid).child("files").shallow().get().val())
        # img_num = len(all_img) + 1
        
        
        
        x = random.randint(0,100)
        while(x in all_img):
            x = random.randint(0,100)
    except:
        print("himlo mc")
        x = random.randint(0,100)

    database.child("users").child(uid).child("files").child(x).set(url)
    return render(request,'community.html')

def myphotos(request):
    global user
    uid = user['localId']
    img_lis = []
    all_img = list(database.child('users').child(uid).child("files").shallow().get().val())
    for i in all_img:
        img = database.child('users').child(uid).child('files').child(i).get().val()
        img_lis.append(img)
    dp = database.child('users').child(uid).child("details").child('dp').get().val()
    return render(request,'my_photos.html',{'img_lis':img_lis , 'dp':dp})    


#to set dp 
def setdp(request):
    global user
    uid = user['localId']
    url = request.POST.get('url')
    database.child("users").child(uid).child("details").child('dp').set(url)
    return render(request,'community.html')

def classi(request):
    return render(request,'classi.html')

def setclassi(request):
    url = request.POST.get('url')
    print(url)
    database.child("users").child('classi_photo').set(url)

    import gc
    gc.collect()
    import pandas as pd
    import numpy as np
    import os
    fpath = r"C:\Users\himal\OneDrive\Desktop\mysite\Data\Images"
    print(os.listdir(fpath))
    dog_classes = os.listdir(fpath)

    # In[2]:

    breeds = [breed.split('-', 1)[1] for breed in dog_classes]
    breeds[:10]

    # In[41]:

    from itertools import chain
    X = []
    y = []
    fullpaths = [r'C:\Users\himal\OneDrive\Desktop\mysite\Data\Images\{}'.format(dog_class) for dog_class in
                 dog_classes]

    for counter, fullpath in enumerate(fullpaths):
        for imgname in os.listdir(fullpath):
            X.append([fullpath + '\\' + imgname])
            y.append(breeds[counter])
    print(X[:10], "\n")
    print(y[:10], "\n")

    X = list(chain.from_iterable(X))
    print(X[:10], "\n")
    len(X)

    # In[42]:

    import random
    combined = list(zip(X, y))
    # print(combined[:10], "\n")
    random.shuffle(combined)
    # print(combined[:10], "\n")
    X[:], y[:] = zip(*combined)

    # In[51]:

    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    # plt.figure(figsize=(18, 18))
    for counter, i in enumerate(random.sample(range(0, len(X)), 9)):
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
    # print(y_ohe.shape)
    y_ohe = np.array(y_ohe)

    # In[54]:

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    img_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in X])
    # print(img_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(img_data, y_ohe, test_size=0.2, random_state=2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2)
    import gc
    del img_data
    gc.collect()

    # In[55]:

    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    batch_size = 32
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=batch_size, seed=1)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_datagerator = val_datagen.flow(x_val, y_val, shuffle=False, batch_size=batch_size, seed=1)

    # In[56]:

    img_id = 2
    dog_generator = train_datagen.flow(x_train[img_id:img_id + 1], y_train[img_id:img_id + 1],
                                       shuffle=False, batch_size=batch_size, seed=1)
    # plt.figure(figsize=(20, 20))
    dogs = [next(dog_generator) for i in range(0, 5)]
    for counter, dog in enumerate(dogs):
        plt.subplot(1, 5, counter + 1)

    # In[57]:

    from keras import models
    from keras import layers
    from keras.optimizers import Adam
    from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
    from keras.applications.inception_v3 import InceptionV3
    from keras.utils.np_utils import to_categorical
    from keras.utils.vis_utils import plot_model

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    model = models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(breeds), activation='softmax'))
    model.layers[0].trainable = False

    #

    # In[58]:

    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # In[ ]:

    # In[59]:

    import pickle
    urllib.request.urlretrieve(url, "temp.png")
    img = 'C:\\Users\\himal\\OneDrive\\Desktop\\mysite\\temp.png'

    model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
    img_data = np.array([img_to_array(load_img(img, target_size = (299,299,3)))])

    # print(img_data)

    # le = LabelEncoder()
    # test_predictions = model.predict(x_test1)
    # predictions = le.classes_[np.argmax(test_predictions, axis=1)]
    # model = pickle.load(open(r'C:\Users\himal\OneDrive\Desktop\mysite\model\finalized_model.sav', 'rb'))
    x_test1 = img_data / 255.
    test_predictions = model.predict(x_test1)
    # print(test_predictions)
    predictions = le.classes_[np.argmax(test_predictions, axis=1)]
    # print(predictions)
    name = predictions[0].upper().replace("_"," ")

    return render(request,'predict.html',{'name':name,'dg':url,'lk':"https://simple.wikipedia.org/wiki/"+predictions[0]})

    # return render(request,'predict.html',{'url':url}
