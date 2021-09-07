from __future__ import print_function

import os
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.patches as mpatches
from scipy.stats import spearmanr


def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

def normalization_for_NE(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 1000000
    features = np.log2(features + 1)
    return features

def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W

def getNeMatrix(W_in):
    N = len(W_in)

    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W


"""
Construct a graph based on the cell features
"""
def getGraph(dataset_str, features, L, K, method):
    print(method)

    if method == 'pearson':
        co_matrix = np.corrcoef(features)
    elif method == 'spearman':
        co_matrix, _ = spearmanr(features.T)
    elif method == 'NE':
        co_matrix = np.corrcoef(features)
        
        NE_path = 'result/NE_' + dataset_str + '.csv'
        if os.path.exists(NE_path):
            NE_matrix = pd.read_csv(NE_path).values
        else:
            features = normalization_for_NE(features)
            in_matrix = np.corrcoef(features)
            NE_matrix = getNeMatrix(in_matrix)
            pd.DataFrame(NE_matrix).to_csv(NE_path, index=False)

        N = len(co_matrix)
        sim_sh = 1.
        for i in range(len(NE_matrix)):
            NE_matrix[i][i] = sim_sh * max(NE_matrix[i])
        
        data = NE_matrix.reshape(-1)
        data = np.sort(data)
        data = data[:-int(len(data)*0.02)]
        
        min_sh = data[0]
        max_sh = data[-1]
        
        delta = (max_sh - min_sh) / 100
    
        temp_cnt = []
        for i in range(20):
            s_sh = min_sh + delta * i
            e_sh = s_sh + delta
            temp_data = data[data > s_sh]
            temp_data = temp_data[temp_data < e_sh]
            temp_cnt.append([(s_sh + e_sh)/2, len(temp_data)])
        
        candi_sh = -1
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1:
                if pear_cnt < temp_cnt[i+1][1] and pear_cnt < temp_cnt[i-1][1]:
                    candi_sh = pear_sh
                    break
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]
                if pear_cnt * 2 < temp_cnt[i-1][1]:
                    candi_sh = pear_sh

        if candi_sh == -1:
            candi_sh = 0.3
        
        print('thres', candi_sh)
        propor = len(NE_matrix[NE_matrix <= candi_sh])/(len(NE_matrix)**2)
        print('propor', propor)

        propor = 1 - propor
        thres = np.sort(NE_matrix)[:, -int(len(NE_matrix)*propor)]
        co_matrix.T[NE_matrix.T <= thres] = 0
            
    else:
        return

    N = len(co_matrix)
    
    up_K = np.sort(co_matrix)[:,-K]
    
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    
    thres_L = np.sort(co_matrix.flatten())[-int(((N*N)//(1//(L+1e-8))))]
    mat_K.T[co_matrix.T < thres_L] = 0

    return mat_K



"""
Load scRNA-seq data set and perfrom preprocessing
"""
def load_data(data_path, dataset_str, PCA_dim, is_NE=True, n_clusters=20, K=None):
    # Get data
    DATA_PATH = data_path

    data = pd.read_csv(DATA_PATH, index_col=0, sep='\t')
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T

    # Preprocess features
    features = normalization(features)

    # Construct graph
    N = len(cells)
    avg_N = N // n_clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)

    L = 0
    if is_NE:
        method = 'NE'
    else:
        method = 'pearson'
    adj = getGraph(dataset_str, features, L, K, method)

    #features = normalization(features)
    # feature tranformation
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components = PCA_dim)
        features = pca.fit_transform(features)
    else:
        var = np.var(features, axis=0)
        min_var = np.sort(var)[-1 * PCA_dim]
        features = features.T[var >= min_var].T
        features = features[:, :PCA_dim]
    print('Shape after transformation:', features.shape)
    
    features = (features - np.mean(features)) / (np.std(features))
  
    return adj, features, cells, genes



def saveClusterResult(y_pred, cells, dataset_str):
    pred_path = 'result/pred_'+dataset_str+'.txt'

    result = []
    for i in range(len(y_pred)):
        result.append([cells[i], y_pred[i]])
    result = pd.DataFrame(np.array(result), columns=['cell', 'label'])
    result.to_csv(pred_path, index=False, sep='\t')


def my_kmeans(K, hidden, dataset_str):
    print('--------------------------------')
    print('Kmeans start, with data shape of', hidden.shape)

    pred_path = 'pred/pretrain_'+dataset_str+'.txt'

    kmeans = KMeans(n_clusters=K, random_state=0).fit(hidden)
    labels = kmeans.labels_

    print('Kmeans end')
    print('--------------------------------')
    return kmeans.labels_, kmeans.cluster_centers_
