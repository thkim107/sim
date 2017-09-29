import h5py
from scipy.spatial import distance
import numpy as np 

path = './Desktop/COVER_SONG/chroma_data_training/CP_1000ms_training_s2113_d30000_170106223401.h5'

f1 = h5py.File(path)
datasetNames=[n for n in f1.keys()]

X = f1['X']
idxDis_train = f1['idxDis_train']
idxDis_validate = f1['idxDis_validate']
idxSim_train = f1['idxSim_train']
idxSim_validate = f1['idxSim_validate']


def simple_matrix(X,Y):
	M = [[0 for col in range(180)] for row in range(180)]
	for i in range(180):
		for j in range(180):
			M[i][j] = distance.euclidean(X[i,:],Y[j,:])
	return np.asarray(M)

 

for i in range(1175):
	for j in range(1175):
		np.save('./Desktop/KAKAO_ALL_PAIR/'+'{:0=4}'.format(i)+'_'+'{:0=4}'.format(j)+'.npy',simple_matrix(X[i],X[j]))
	print((str)(i)+'th complete')
