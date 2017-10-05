import h5py
from scipy.spatial import distance
import scipy.misc
import numpy as np 
from joblib import Parallel, delayed  # installation by 'conda install joblib'

#path = '/home/sungkyun/Dropbox/kakao coversong git/sim/data/training/CP_1000ms_training_s2113_d2113_170106223452.h5'
path = '/home/sungkyun/Dropbox/kakao coversong git/sim/data/eval/Eval1000_CP_1000ms.h5'

f1 = h5py.File(path)
datasetNames=[n for n in f1.keys()]

X = f1['X']

#%%

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



#%% Preprocess & Save Eval data



def my_func(idx_start):
    for i in range(idx_start, idx_start+1):
        print((str)(i)+'th start processing')
        for j in range(1000):
            scipy.misc.imsave('/home/sungkyun/Data/KAKAO_ALL_PAIR_EVAL/'+'{:0=4}'.format(i)+'_'+'{:0=4}'.format(j)+'.jpg',simple_matrix(X[i],X[j]))
    
        print((str)(i)+'th complete')
    return 0




#%% multithread : using 7 thread 
idx_start=range(0,330)
n_thread = -1
_ = Parallel(n_jobs=n_thread, verbose=10, backend="multiprocessing")(map(delayed(my_func), idx_start ))
