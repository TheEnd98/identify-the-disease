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
import cv2
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
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


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
#print(CNN.summary())# in các thông số của mạng 


#load trọng số của mạng
CNN.load_weights('model.hdf5')



#đưa dữ liệu để test
Data= []
test = os.listdir("/home/dungnguyen/test")
for x in test: # For every uninfected Picture
    Data.append(["/home/dungnguyen/test/"+x,0])
Image = [x[0] for x in Data]
del Data
image = []
X_test = []
image=Image
def GetPic(path):
    im = cv2.imread(path,1)
    im = cv2.resize(im,(64,64)) 
    im = im/255
    return im
for x in range(len(image)):
    X_test.append(GetPic(image[x]))
X_test=np.array(X_test)
predictions = CNN.predict(X_test)

for i in range(len(X_test)):
	fig=plt.figure(figsize=(10,10))
	fig.add_subplot(1,1,1)
	plt.imshow(X_test[i])
	if np.around(predictions[i], decimals=0)== [0]:
		plt.title('Ty le nhiem benh: {:.2%}\n KL:KHONG Nhiem benh\n'.format(float(predictions[i])))
	else:
		plt.title('Ty le nhiem benh: {:.2%}\n KL:BI Nhiem benh\n'.format(float(predictions[i])))
	plt.show()

