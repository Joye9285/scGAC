from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
from keras.engine.topology import Layer, InputSpec

import tensorflow as tf


class ClusteringLayer(Layer):
    
    def __init__(self,
                 n_clusters,
                 N,
                 mode='q2',
                 sh = .1,
                 weights=None,
                 alpha=1.0,
                 **kwargs):
        self.n_clusters = n_clusters
        self.N = N
        self.sh = sh
        self.mode = mode
        self.alpha = alpha
        self.clusters = weights

        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.built = True

    def call(self, inputs, **kwargs):
        # q_ik = (1 + ||z_i - miu_k||^2)^-1
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        # q_ik = q_ik / sigma_k' q_ik'
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
            
        # q'_ik = q_ik ^ 2 / sigma_i q_ik
        if self.mode == 'q2':
            q_ = q ** 2 / K.sum(q, axis=0)
            q_ = K.transpose(K.transpose(q_) / K.sum(q_, axis=1))
        else:
            q_ = q + 1e-20
        
        q_idx = K.argmax(q_, axis=1)
        q_mask = K.one_hot(q_idx, self.n_clusters)
        # q'_ik = 0 if q'_ik < max(q'_i)
        q_ = q_mask * q_
         
        q_ = K.relu(q_ - self.sh)
        q_ = q_ + K.sign(q_) * self.sh
        # miu_k = sigma_i q'_ik * z_i
        self.clusters = K.dot(K.transpose(q_ / K.sum(q_, axis=0)), inputs)
       
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
