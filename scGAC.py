from __future__ import division

import pickle as pkl
import numpy as np
import pandas as pd
import sys
import time
import os
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy
import scipy.stats

from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras import backend as K
import keras

from graph_attention_layer import GraphAttention
from soft_clustering_layer import ClusteringLayer
from utils import load_data, my_kmeans, saveClusterResult, saveResultLog


ARGV_L = [0]
k = 16
ARGV_DR = [0.4]
L1 = 0
LR = 5e-4
ARGV_BETA = [0.]
ARGV_DC = [0.]
ARGV_GM = [1.]
ARGV_Cs = [[0., 1., 1., 0., 0.]]#, 
t = 2
ARGV_Q_SH = [0]
ARGV_CR = ['NE']
ARGV_NESH = [0.5]
ARGV_Q_MODE = ['q2']


pre_lr = 2e-4
pre_ep = 200
pre_A = [pre_lr, pre_ep]


ARGVs = []
ARGVs.append([0, k, 64, 16, 4, 0.4, 
              L1, LR, 0., 0., 1., 
              0., 1., 1., 0., 0., t, 0,
              'NE', 0.5, 'inner'])

param = [True,  5,  True] + ['zscore']
ARGV = ARGVs[0]
trans = 'ex-1'
PCA_dim = 512
n_clusters = int(sys.argv[4])
# Parameters
L = 0     # L percent of nodes will be focused on
k = 16    # K neighbors will be focused on at least
dataset_str = sys.argv[1]#'GSE70580'      # Dataset name
cr_method = 'NE'
NEsh = 0.5
q_mode = ARGV[20]

# paths
GAT_autoencoder_path = 'logs/GATae_'+dataset_str+'.h5'
model_path = 'logs/model_'++dataset_str+'.h5'
pred_path = 'result/pred_'+dataset_str+'.txt'
true_path = 'data/'+dataset_str+'/subtype.ann'
intermediate_path = 'logs/model_'++dataset_str+'_'

# Read data
start_time = time.time()
A, X, rowsum, cells, genes = load_data(dataset_str, L, k, param, 
                                       PCA_dim, cr_method, NEsh, 
                                       n_clusters)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-process: run time is %.2f '%run_time, 'minutes')



# Parameters
N = X.shape[0]                  # Number of nodes in the graph
F = X.shape[1]                  # Original feature dimension
F1 = int(ARGV[2])#64            # Output size of first GAL
F2 = int(ARGV[3])#16
n_attn_heads = int(ARGV[4])     # Number of attention heads
dropout_rate = float(ARGV[5])   # Dropout rate
l2_reg = float(ARGV[6])#1e-6    # Factor for l2 regularization
learning_rate = float(ARGV[7])  # Learning rate for Adam
pre_lr = pre_A[0]#1e-4
pre_epochs = pre_A[1]#200#100#int(ARGV[9])#500   # Number of training epochs
epochs = 5000#1000
es_patience = 50#int(ARGV[10])  # Patience for early stopping
es_delta = 0.1                  # Min delta for early stopping
beta = float(ARGV[8])
dc = float(ARGV[9])
gamma = float(ARGV[10])
c1 = float(ARGV[11])#1.#1e-4
c2 = float(ARGV[12])#0#1.
c3 = float(ARGV[13])
c4 = float(ARGV[14])
c5 = float(ARGV[15])
update_interval = int(ARGV[16])
paint_interval = 500
q_sh = float(ARGV[17])
cr_method = ARGV[18]

# Update loss function
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true))

def DAEGC_class_loss_1(y_pred):
    return K.mean(K.exp(-1 * A * K.sigmoid(K.dot(y_pred, K.transpose(y_pred)))))

# Total loss = mean_absolute_imputation_error + class_loss
def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    loss_C = DAEGC_class_loss_1(y_pred)
    return c1 * loss_C + c2 * loss_E


# Model definition (as per Section 3.3 of the paper)
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(F1,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   beta=beta,
                                   gamma=gamma,
                                   trans=trans,
                                   activation='elu',
                                   kernel_regularizer=l1(l2_reg),
                                   #bias_regularizer=l1(l2_reg),
                                   attn_kernel_regularizer=l1(l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(F2,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   beta=beta,
                                   gamma=gamma,
                                   trans=trans,
                                   activation='elu',
                                   kernel_regularizer=l1(l2_reg),
                                   #bias_regularizer=l1(l2_reg),
                                   attn_kernel_regularizer=l1(l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(F1,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   beta=beta,
                                   gamma=gamma,
                                   trans=trans,
                                   activation='elu',
                                   kernel_regularizer=l1(l2_reg),
                                   #bias_regularizer=l1(l2_reg),
                                   attn_kernel_regularizer=l1(l2_reg))([dropout3, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_3)
graph_attention_4 = GraphAttention(F,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   beta=beta,
                                   gamma=gamma,
                                   trans=trans,
                                   activation='elu',
                                   kernel_regularizer=l1(l2_reg),
                                   #bias_regularizer=l1(l2_reg),
                                   attn_kernel_regularizer=l1(l2_reg))([dropout4, A_in])

# Build GAT autoencoder model
GAT_autoencoder = Model(inputs=[X_in, A_in], outputs=graph_attention_4)
optimizer = Adam(lr=pre_lr)
GAT_autoencoder.compile(optimizer=optimizer,
              loss=maie_class_loss)
#GAT_autoencoder.summary()

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=es_delta, patience=es_patience)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(GAT_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train GAT_autoencoder model
start_time = time.time()
GAT_autoencoder.fit([X, A],X,epochs=pre_epochs,batch_size=N,
                    verbose=0,shuffle=False)#,
                    #callbacks=[es_callback, tb_callback, mc_callback])
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-train: run time is %.2f '%run_time, 'minutes')


# Construct a model for hidden layer
hidden_model = Model(inputs=GAT_autoencoder.input,outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)


# Get k-means clustering results of hidden representation of cells
y_pred, pre_centers = my_kmeans(n_clusters, hidden, dataset_str)
y_pred_last = np.copy(y_pred)


# Add the soft_clustering layer
soft_cluster_layer = ClusteringLayer(n_clusters,
                                     N,
                                     q_mode,#'q2',
                                     q_sh,#0,
                                     pre_centers,
                                     name='clustering')(dropout3)

def pred_loss(y_true, y_pred):
    return y_pred

# Construct total model
model = Model(inputs=[X_in, A_in],
              outputs=[graph_attention_4,
                       soft_cluster_layer,
                       graph_attention_2])


optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss=[maie_class_loss, 'kld', pred_loss],
                    loss_weights=[c2, c3, 0])


# Train model
start_time = time.time()

tol = 1e-5
loss = 0

pre_ARI = -1
end_ARI = -1
sil_logs = []
es_thres = 0.02
es_epochs = 100
n_epoch = int(es_epochs // update_interval)
res_ite = 0

for ite in range(epochs + 1):
    if ite % update_interval == 0:
        res_ite = ite

        _, q, hid = model.predict([X, A], batch_size=N, verbose=0)
        p = q ** 2 / q.sum(0)
        p = (p.T / p.sum(1)).T
        y_pred = p.argmax(1)
        
        sil_hid = metrics.silhouette_score(hid, y_pred, metric='euclidean')
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        print('Iter:', ite, 
              ', sil_hid:', np.round(sil_hid, 3),
              ', delta_label', np.round(delta_label, 3), 
              ', loss:', np.round(loss, 2))

        sil_logs.append(sil_hid)
        arr_sil = np.array(sil_logs)
        if len(arr_sil) >= 30 * 2:
            mean_0_n = np.mean(arr_sil[-30:])
            mean_n_2n = np.mean(arr_sil[-60: -30])
            if mean_0_n - mean_n_2n <= es_thres:
                print('Stop early at', ite, 'epoch')
                break
        
        saveClusterResult(y_pred, cells, dataset_str)

        if pre_ARI == -1:
            pre_ARI = ARI
        end_ARI = ARI
        end_NMI = NMI
        
    loss = model.train_on_batch(x=[X, A], y=[X, p, hid])


model.save_weights(model_path)

end_time = time.time()
run_time = (end_time - start_time) / 60
print('Train: run time is %.2f '%run_time, 'minutes')

saveClusterResult(y_pred, cells, dataset_str)


# Get hidden representation
hidden_model = Model(inputs=model.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)

mid_str = dataset_str
hidden = pd.DataFrame(hidden)
hidden.to_csv('result/hidden_'+mid_str+'.tsv', sep='\t')


print('Done.')
