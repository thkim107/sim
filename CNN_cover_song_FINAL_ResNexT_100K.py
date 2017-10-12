import numpy as np 
import glob
import os
import re
import h5py
import scipy.misc
from keras import models
from keras import layers
from keras.utils import np_utils
import gc # garbage collect
# NOTE:
# - replace data path (line:19~22)
# - replace save path (line 217, 236)
# - line 238~end is for evalutaion. you can remove it.

# options
np.random.seed(777)
# to use 30K, 100K, change h5 only
path_train_h5 = './data/training/CP_1000ms_training_s2113_d100000_170106223159.h5' 
path_train_simmtx = '/home/sungkyun/Data/KAKAO_ALL_PAIR_TRAIN_100K/' 
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

X_mem = [] # training data, we use disk as memory for large X
X = []
Y = [] # training data
Xval = [] # validation data
Yval = [] # validation data

# Aggregate Training data
for i in range(np.shape(idxSim_train)[0]):
	a=[idxSim_train[i][0], idxSim_train[i][1]]
	X_mem.append(scipy.misc.imread(path_train_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_S.jpg'))
	Y.append(1)

for i in range(np.shape(idxDis_train)[0]): #for i in range(np.shape(idxDis_train)[0]-20000):
	a=[idxDis_train[i][0], idxDis_train[i][1]]
	X_mem.append(scipy.misc.imread(path_train_simmtx +'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_D.jpg'))
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


#%% how to use disk as memory? 
# a = np.memmap('test.mymemmap', dtype='float32', mode='w+', shape = (4226,180,180,1))
X = np.memmap('temp.mymemmap', dtype='float32', mode='w+', shape = (len(X_mem),180,180, 1))
for i in range(0, len(X_mem)):
    X[i,:,:,0] = np.float32(X_mem[i])
    print(i)

del(X_mem)
gc.collect()

#X/=np.max(X)
Y = np_utils.to_categorical(Y,2)

Xval = np.asarray(Xval)
Xval = Xval.reshape(Xval.shape[0], 180, 180, 1)
Xval = Xval.astype('float32')
#X/=np.max(X)
Yval = np_utils.to_categorical(Yval,2)

# Feature scaling:  sc_factor = [[mu], [std]]
sc_factor = [np.mean(X), np.std(X[0:2113,:,:,0])]  # np.std is unstable with large array now.

for i in range(0, X.shape[0]):
    X[i,:,:,0] = (X[i,:,:,0] - sc_factor[0]) / sc_factor[1]
    print('Feature Scaling: ', i,'/',X.shape[0])

Xval = (Xval-sc_factor[0]) / sc_factor[1]

gc.collect()
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
#model.summary()
#model.count_params()
#Total params: 23,080,066
#Trainable params: 23,011,842
#Non-trainable params: 68,224

#%%
model.fit(X, Y, batch_size=4, epochs=30, verbose=1, validation_data=(Xval, Yval), shuffle=True)
#%% save: You can save only weights, due to Keras' bug on LeakyReLU

model.save_weights('/home/sungkyun/Data/saved_models/ResNexT1_100K_weights_ep30_.h5')
#model.load_weights('/home/sungkyun/Data/saved_models/ResNexT1_100K_weights_ep30_.h5', by_name=False)

#%% predict
cover_p_mtx = np.zeros([1000, 1000])
noncover_p_mtx = np.zeros([1000, 1000])

for i in range(0, 1000):
    for j in range(0,1000): 
        # xsim_mtx: cross similarity matrix that compares song i and j
        xsim_mtx = np.ndarray([1, 180, 180, 1], dtype='float32')
        xsim_mtx[0,:,:,0] = scipy.misc.imread(path_eval_simmtx + '{:0=4}'.format(np.min([i,j]))+'_'+'{:0=4}'.format(np.max([i,j]))+'.jpg')
        xsim_mtx = (xsim_mtx - sc_factor[0]) / sc_factor[1] # feature scaling with zero mean unit standardization
        print([i,j])
        
        # prediction
        cover_p_mtx[i, j] = model.predict(xsim_mtx)[0][1]
        noncover_p_mtx[i, j] = model.predict(xsim_mtx)[0][0]
                
np.save('/home/sungkyun/Data/saved_models/cover_p_mtx_ResNexT1_100K_weights_ep30.npy', cover_p_mtx)


#%% ranking method
# I. ranking by maximum cover probability: r_max_prob
r_max_prob = np.argsort(-cover_p_mtx)[0:330, 0:10] # sort: idx of larger number places left 

# II. ranking by representation vector with cosine distance: r_repvec_cos
from scipy import spatial
repvec_cos = np.zeros([330,1000])  # fill with very large numbers
        
for i in range(0,330):
    for j in range(1000):
        if i==j:
            repvec_cos[i,j] = 1
        else:
            repvec_cos[i,j] = spatial.distance.cosine(cover_p_mtx[i,:], cover_p_mtx[j,:])
      
r_repvec_cos = np.argsort(repvec_cos)[:, 0:10] # sort: idx of smaller number places left 
#%% Evaluation
# input_ranking must be 330 x 1000 matrix
# Eval rank method 1
input_ranking = r_max_prob

score = 0
for i in range(0, 330, 11):
    score += np.sum(np.sum((i<=input_ranking[i:i+11,:]) & (input_ranking[i:i+11,:]<=i+10), axis=1), axis=0)

print('rank method1 score =',score)

# Eval rank method 2
input_ranking = r_repvec_cos 
score = 0
for i in range(0, 330, 11):
    score += np.sum(np.sum((i<=input_ranking[i:i+11,:]) & (input_ranking[i:i+11,:]<=i+10), axis=1), axis=0)

print('rank method1 score =',score)

    
    
#%%
#from matplotlib import pyplot as plt
#plt.imshow(r_max_prob, interpolation='nearest')
#plt.show()



