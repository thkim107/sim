#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data.py

Created on Tue Aug 15 18:39:48 2017

"""

import numpy as np
import h5py





def preprocess_YH_data():
    
    #%% Read files
    print('read files...')
    with h5py.File('./data/training/CP_1000ms_training_s2113_d2113_170106223452.h5','r') as f:
        X = np.asarray(f['X'])
        idxDis_tr = np.asarray(f['idxDis_train']).astype(int)-1     # -1 because matlab idx start from 1
        idxDis_val = np.asarray(f['idxDis_validate']).astype(int)-1
        idxSim_tr = np.asarray(f['idxSim_train']).astype(int)-1
        idxSim_val = np.asarray(f['idxSim_validate']).astype(int)-1



    #%% dim is feature dimension, tdim is time dimension
    tdim = X.shape[1]
    dim = X.shape[2] 


    #%% read indices and put data in X_train and X_test
    X_train = np.zeros((idxSim_tr.shape[0]+idxDis_tr.shape[0],2,tdim,dim))  
    X_train[0:idxSim_tr.shape[0],0,:,:] =  X[idxSim_tr[:,0],:,:]
    X_train[0:idxSim_tr.shape[0],1,:,:] =  X[idxSim_tr[:,1],:,:]
    X_train[idxSim_tr.shape[0]:,0,:,:] =  X[idxDis_tr[:,0],:,:]
    X_train[idxSim_tr.shape[0]:,1,:,:] =  X[idxDis_tr[:,1],:,:]
    
    X_test = np.zeros((idxSim_val.shape[0]+idxDis_val.shape[0],2,tdim,dim))
    X_test[0:idxSim_val.shape[0],0,:,:] =  X[idxSim_val[:,0],:,:]
    X_test[0:idxSim_val.shape[0],1,:,:] =  X[idxSim_val[:,1],:,:]
    X_test[idxSim_val.shape[0]:,0,:,:] =  X[idxDis_val[:,0],:,:]
    X_test[idxSim_val.shape[0]:,1,:,:] =  X[idxDis_val[:,1],:,:]
    #%%
    
    return
    
    


# Optimal transposition index (OTI) implementation: returns transposed chromas
def oti(cover1,cover2,chroma_dim): 
    cover1_mean = np.sum(cover1,axis=0)/np.max(np.sum(cover1,axis=0)) # get global HPCP first
    cover2_mean = np.sum(cover2,axis=0)/np.max(np.sum(cover2,axis=0))
    dist_store = np.zeros(chroma_dim)
    
    for i in range(0,chroma_dim):
        cover2_mean_shifted = np.roll(cover2_mean, i) # circular shift
        dist = np.dot(cover1_mean,cover2_mean_shifted) # get dot product
        dist_store[i] = dist
        
    oti = np.argmax(dist_store) # how many shift was optimal
    cover2_shifted = np.roll(cover2, oti, axis=1) # to return shifted chroma
    
    return cover1, cover2_shifted

