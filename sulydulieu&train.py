import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import fnmatch
import keras
from time import sleep
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as k
import random
from keras.models import load_model


Data = []
Uninfected = os.listdir("/home/dungnguyen/hinhanhsotret/cell_images/Uninfected")
Parasitized = os.listdir("/home/dungnguyen/hinhanhsotret/cell_images/Parasitized")

for x in Uninfected: # For every uninfected Picture
    Data.append(["/home/dungnguyen/hinhanhsotret/cell_images/Uninfected/"+x,0]) # dán nhãn 0 cho tập Uninfected
    
for x in Parasitized: #For every infected Picture
    Data.append(["/home/dungnguyen/hinhanhsotret/cell_images/Parasitized/"+x,1]) # dán nhãn 1 cho tập infected
    
    
random.shuffle(Data) # trộn tập dữ liệu


Image = [x[0] for x in Data] # bao gồm các hình ảnh 
Label = [x[1] for x in Data] # bao gồm các nhãn (ứng với hình ảnh)

del Data


X_train, X_test, Y_train, Y_test = train_test_split(Image, Label, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=46) # tách dữ liệu thành các dữ liệu thử nghiệm



def GetPic(path):
    im = cv2.imread(path,1)
    im = cv2.resize(im,(64,64)) # cắt thành ảnh có kích thước phù hợp
    im = im/255
    return im


X_images = []
Y_images = []
X_val_im = []
Y_val_im = []


c = 0

for x in range(len(X_train)):

    try:
        X_images.append(GetPic(X_train[x]))
        Y_images.append(Y_train[x])
        c += 1
    
    except:
        print('c: ' + str(c))

        
Y_train = Y_images# tập dữ liệu để train 


c = 0

for x in range(len(X_val)): 

    try:
        X_val_im.append(GetPic(X_val[x]))
        Y_val_im.append(Y_val[x])
    
    except:
        print('c: ' + str(c))
        
Y_val = Y_val_im # một phần của dữ liệu để đánh giá hàm lost


X_images = np.array(X_images)
X_val_im = np.array(X_val_im)




CNN = Sequential() # tạo một mạng mới 
CNN.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(64,64,3)))
CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

CNN.add(Conv2D(32, kernel_size=3, activation='relu'))
CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

CNN.add(Conv2D(32, kernel_size=3, activation='relu')) 
CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

CNN.add(Flatten())
CNN.add(Dense(128, activation='sigmoid'))

CNN.add(Dense(1, activation='sigmoid'))

CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mse','mae'])
print(CNN.summary())# in các thông số của mạng 

History = CNN.fit(X_images, Y_train, validation_data=(X_val_im, Y_val), epochs = 5)# tiến hành training 

model_json = CNN.to_json()# lưu lại mô hình 
with open("model.json", "w") as json_file:
    json_file.write(model_json)
CNN.save_weights("model.hdf5")
print("Saved model to disk")


fig = plt.figure()                                          #hiển thị kết quả học tập 
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(History.history['acc'], color='red')
ax.plot(History.history['val_acc'], color ='black')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


fig = plt.figure()                                          #hiển thị tỉ lệ lỗi 
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(History.history['loss'], color='red')
ax.plot(History.history['val_loss'], color ='black')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()




CNN.load_weights('model.hdf5')                             #load lại model cho các lần sau 
