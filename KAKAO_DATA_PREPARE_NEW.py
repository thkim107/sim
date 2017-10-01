import h5py
from scipy.spatial import distance
import scipy.misc
import numpy as np 

path = './Desktop/COVER_SONG/chroma_data_training/CP_1000ms_training_s2113_d2113_170106223452.h5'

f1 = h5py.File(path)
datasetNames=[n for n in f1.keys()]

X = f1['X']
idxDis_train = f1['idxDis_train']
idxDis_validate = f1['idxDis_validate']
idxSim_train = f1['idxSim_train']
idxSim_validate = f1['idxSim_validate']


def oti(cover1,cover2,chroma_dim): 
    cover1_mean = np.sum(cover1,axis=0)/np.max(np.sum(cover1,axis=0)) 
    cover2_mean = np.sum(cover2,axis=0)/np.max(np.sum(cover2,axis=0))
    dist_store = np.zeros(chroma_dim)
    for i in range(0,chroma_dim):
        cover2_mean_shifted = np.roll(cover2_mean, i) 
        dist = np.dot(cover1_mean,cover2_mean_shifted) 
        dist_store[i] = dist 
    oti = np.argmax(dist_store)
    cover2_shifted = np.roll(cover2, oti, axis=1)
    return cover1, cover2_shifted



def simple_matrix(X,Y):
	XX = oti(X,Y,12)[0]
	YY = oti(X,Y,12)[1]
	M = [[0 for col in range(180)] for row in range(180)]
	for i in range(180):
		for j in range(180):
			M[i][j] = distance.euclidean(XX[i,:],YY[j,:])
	return np.asarray(M)




# np.shape(idxSim_train)[0]
for i in range(np.shape(idxSim_train)[0]):
	a=[idxSim_train[i][0], idxSim_train[i][1]]
	scipy.misc.imsave('./Desktop/KAKAO_ALL_PAIR_TRAIN/'+'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_S.jpg',simple_matrix(X[min(a)-1],X[max(a)-1]))
		
	print((str)(i)+'th complete')

# np.shape(idxDis_train)[0]
for i in range(np.shape(idxDis_train)[0]):
	a=[idxDis_train[i][0], idxDis_train[i][1]]
	scipy.misc.imsave('./Desktop/KAKAO_ALL_PAIR_TRAIN/'+'{:0=4}'.format((int)(min(a)))+'_'+'{:0=4}'.format((int)(max(a)))+'_D.jpg',simple_matrix(X[min(a)-1],X[max(a)-1]))
		
	print((str)(i)+'th complete')


# 1175 x 1175 pair (180 by 180 matrix) complete


