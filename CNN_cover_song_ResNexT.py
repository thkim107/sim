import numpy as np 
import glob
import os
import re
import h5py
import scipy.misc
from keras import models
from keras import layers
from keras.utils import np_utils

# options
np.random.seed(777)
# to use 30K, 100K, change h5 only
path_train_h5 = './data/training/CP_1000ms_training_s2113_d30000_170106223401.h5' 
path_train_simmtx = '/home/sungkyun/Data/KAKAO_ALL_PAIR_TRAIN_30K/' 
path_val_simmtx = '/home/sungkyun/Data/KAKAO_ALL_PAIR_VAL/'
path_eval_simmtx = '/home/sungkyun/Data/KAKAO_ALL_PAIR_EVAL/'

# ResNexT parameters
cardinality = 32

#%% read paired list from h5 file
f1 = h5py.File(path_train_h5)
datasetNames=[n for n in f1.keys()]

FX = f1['X']
idxDis_train = f1['idxDis_train']
idxDis_validate = f1['idxDis_validate']
idxSim_train = f1['idxSim_train']
idxSim_validate = f1['idxSim_validate']

X = [] # training data
Y = [] # training data
Xval = [] # validation data
Yval = [] # validation data

# Aggregate Training data
for i in range(np.shape(idxSim_train)[0]):
	a=[idxSim_train[i][0], idxSim_train[i][1]]
	X.append(scipy.misc.imread(path_train_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_S.jpg'))
	Y.append(1)

for i in range(2113): #for i in range(np.shape(idxDis_train)[0]-20000):
	a=[idxDis_train[i][0], idxDis_train[i][1]]
	X.append(scipy.misc.imread(path_train_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_D.jpg'))
	Y.append(0)


# Aggregate Validation data
for i in range(np.shape(idxSim_validate)[0]):
    a=[idxSim_validate[i][0], idxSim_validate[i][1]]
    Xval.append(scipy.misc.imread(path_val_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_S.jpg'))
    Yval.append(1)
  
for i in range(np.shape(idxDis_validate)[0]):
	a=[idxDis_validate[i][0], idxDis_validate[i][1]]
	Xval.append(scipy.misc.imread(path_val_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_D.jpg'))
	Yval.append(0)

X = np.asarray(X)
X = X.reshape(X.shape[0], 180, 180, 1)
X = X.astype('float32')
#X/=np.max(X)
Y = np_utils.to_categorical(Y,2)

Xval = np.asarray(Xval)
Xval = Xval.reshape(Xval.shape[0], 180, 180, 1)
Xval = Xval.astype('float32')
#X/=np.max(X)
Yval = np_utils.to_categorical(Yval,2)



# Feature scaling:  sc_factor = [[mu], [std]]
sc_factor = [np.mean(X), np.std(X)]

X = (X - sc_factor[0]) / sc_factor[1]
Xval = (Xval-sc_factor[0]) / sc_factor[1]
#%% ResNexT: code based on https://blog.waya.ai/deep-residual-learning-9610bb62c355


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y
    

    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x) # BN - ReLU

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        # residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False)
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut) 

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(1)(x)
    x = layers.Dense(2, activation='softmax')(x)

    return x

# X <= N x 180 x 180 x 1 data 			(cover pair N1, non-cover pair N2, N1+N2 = N)
# Y <= N x (1 or 0) data as one-hot 	(cover pair N1, non-cover pair N2, N1+N2 = N)
image_tensor = layers.Input(shape=(180,180,1))
network_output = residual_network(image_tensor)
  
model = models.Model(inputs=[image_tensor], outputs=[network_output])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
model.summary()
model.count_params()


#%%
model.fit(X, Y, batch_size=4, epochs=30, verbose=1, validation_data=(Xval, Yval), shuffle=True)
#%% save: You can save only weights, due to Keras' bug on LeakyReLU

model.save_weights('./save/ResNexT1_weights_ep35.h5')
#model.load_weights('./save/my_model_weights1.h5', by_name=False)

#%%

score=[[0 for row in range(1000)] for col in range(330)]
score=np.asarray(score)
for i in range(0,330):
	bb=[]
	for j in range(1000):
		compare=[i,j]
		b = scipy.misc.imread(path_eval_simmtx +'{:0=4}'.format(np.min(compare))+'_'+'{:0=4}'.format(np.max(compare))+'.jpg')
		b = (b - sc_factor[0]) / sc_factor[1] # b=b/np.max(b)
		b=np.asarray(b)
		b=b.reshape(1,180,180,1)
		b=b.astype('float32')
		if i==j: 
			bb.append(0)
		else :	
			bb.append(model.predict(b)[0][1])
	order_b = np.flip(np.argsort(bb),axis=0)
	for k in range(10):
		score[i,order_b[k]]=1
	print((str)(i)+'th_complete')

correct=[[0 for row in range(1000)] for col in range(330)]
correct=np.asarray(correct)
for i in range(30):
	correct[11*i:11*i+11,11*i:11*i+11]=1


result=np.multiply(score,correct)
acc=np.sum(result)
print(acc)
np.savetxt('./save/top10_result_CNN_score1.txt',score)
np.savetxt('./save/top10_result_CNN_accuracy1.txt',acc)

# result
#  
# 2420 @epoch35 (first layer was modified to 3x3)