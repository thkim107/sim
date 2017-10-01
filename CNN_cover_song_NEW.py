# 85% accuracy

import numpy as np 
import glob
import os
import re
import h5py
import scipy.misc

np.random.seed(777)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import LeavePOut


path = './Desktop/COVER_SONG/chroma_data_training/CP_1000ms_training_s2113_d2113_170106223452.h5'

f1 = h5py.File(path)
datasetNames=[n for n in f1.keys()]

FX = f1['X']
idxDis_train = f1['idxDis_train']
idxDis_validate = f1['idxDis_validate']
idxSim_train = f1['idxSim_train']
idxSim_validate = f1['idxSim_validate']

X=[]
Y=[]
for i in range(np.shape(idxSim_train)[0]):
	a=[idxSim_train[i][0], idxSim_train[i][1]]
	X.append(scipy.misc.imread('./Desktop/KAKAO_ALL_PAIR_TRAIN/'+'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_S.jpg'))
	Y.append(1)

for i in range(np.shape(idxDis_train)[0]):
	a=[idxDis_train[i][0], idxDis_train[i][1]]
	X.append(scipy.misc.imread('./Desktop/KAKAO_ALL_PAIR_TRAIN/'+'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_D.jpg'))
	Y.append(0)

X = np.asarray(X)
X = X.reshape(X.shape[0], 180, 180, 1)
X = X.astype('float32')
X/=np.max(X)
Y=np_utils.to_categorical(Y,2)



# X <= N x 180 x 180 x 1 data 			(cover pair N1, non-cover pair N2, N1+N2 = N)
# Y <= N x (1 or 0) data as one-hot 	(cover pair N1, non-cover pair N2, N1+N2 = N)

PP=np.zeros(4226)
YY=np.zeros(4226)
YY[0:2112]=1
seed=7

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
iter=0
score_result = []
for train, test in kfold.split(PP, YY):

	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(5,5), strides=1, padding='valid', activation='relu', input_shape=(180,180,1)))
	model.add(BatchNormalization())
	model.add(Conv2D(32,kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
	model.add(pooling.MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(32,kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
	model.add(Conv2D(16,kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
	model.add(pooling.MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(32,kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
	model.add(Conv2D(16,kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
	model.add(pooling.MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())



	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X[train], Y[train], batch_size=16, nb_epoch=100, verbose=1)


	score = model.evaluate(X[test], Y[test], verbose=0)
	print(model.metrics_names)
	print(score)
	score_result.append(score)
np.savetxt('./Desktop/result2.txt',score_result)
