import numpy as np 
import glob
import os
import re

np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.utils import np_utils

def extract_number(s):
  return int(s.split('/')[-1].split('th')[0])

all_train_files = sorted(glob.glob('./Desktop/KAKAO_TRAIN_DATA/*.npy'), key=extract_number)
all_test_files = sorted(glob.glob('./Desktop/KAKAO_TEST_DATA/*.npy'), key=extract_number)

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

for i in range(np.shape(all_train_files)[0]):
	X_train.append(np.load(all_train_files[i]))
for i in range(np.shape(all_test_files)[0]):
	X_test.append(np.load(all_test_files[i]))


for i in range(1400):
	Y_train.append(1)
for i in range(1400):
	Y_train.append(0)

for i in range(713):
	Y_test.append(1)
for i in range(713):
	Y_test.append(0)


X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
Y_train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)

X_train=X_train.reshape(X_train.shape[0], 180, 180, 1)
X_test=X_test.reshape(X_test.shape[0], 180, 180, 1)

X_train /= np.max(X_train)
X_test /= np.max(X_test)

Y_train = np_utils.to_categorical(Y_train,2)
Y_test = np_utils.to_categorical(Y_test,2)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=2, padding='valid', activation='relu', input_shape=(180,180,1)))
model.add(Conv2D(32,kernel_size=(3,3), strides=2, padding='valid', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=20, verbose=1)


score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)
