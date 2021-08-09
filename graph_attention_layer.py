from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 beta=1.,
                 gamma=1e-8,
                 trans='linear',
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.beta = beta
        self.gamma = gamma
        self.trans = trans
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
            
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        N = K.shape(X)[0]
        F = K.shape(X)[1]

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            F_ = K.shape(kernel)[-1]
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
            
            
            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            # dense = 1/(K.abs(attn_for_self - K.transpose(attn_for_neighs)) + 1e-4)  # (N x N) via broadcasting
            dense = K.exp(-K.square(attn_for_self - K.transpose(attn_for_neighs))/1e-0)
            

            # e ^ ( - (a_1 * W * h_i - a_2 * W * h_j)^2)
            """
            gamma = 1.0
            features = K.dot(X, kernel)
            attn_for_self = features * K.transpose(attention_kernel[0])
            attn_for_neighs = features * K.transpose(attention_kernel[1])
            dense = K.reshape(attn_for_self, (1, N, F_)) - K.reshape(attn_for_neighs, (N, 1, F_))
            dense = K.exp(-K.sum(K.square(dense), axis=-1)/gamma)
            """


            # a * 1 / (W * ||h_i - h_j||)
            """
            features = K.dot(X, kernel)
            M = K.reshape(X, (1, N, F)) - K.reshape(X, (N, 1, F))
            M = K.dot(K.abs(M), kernel)
            M = 1 / (M + 1e-8)
            dense = K.reshape(K.dot(M, attention_kernel[0]), (N, N))
            """

            # a * 1 / (||W * h_i - W * h_j||)
            """
            features = K.dot(X, kernel)
            M = K.reshape(features, (1, N, F_)) - K.reshape(features, (N, 1, F_))
            #M = K.dot(K.abs(M), kernel)
            M = K.abs(M)
            M = K.exp(-M)#1 / (M + 1e-8)
            #dense = K.reshape(K.dot(M, attention_kernel[0]), (N, N))
            dense = K.sum(M, axis=-1)
            """

            # (a_1 * W * h_i) * (a_2 * W * h_j)
            """
            features = K.dot(X, kernel)
            attn_for_self = features * K.transpose(attention_kernel[0])
            attn_for_neighs = features * K.transpose(attention_kernel[1])
            dense = 1-K.exp(-K.square(K.dot(attn_for_self, K.transpose(attn_for_neighs))))
            """

            """
            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)
            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask
            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)
            """
            # Handle the 1, -1, and 0 values, seperately
            mask_0 = K.abs(A)
            mask_pos = 0.5 * (A + 1.) * mask_0
            #mask_neg = 0.5 * (1. - A) * mask_0
            
            gamma = self.gamma
            trans = self.trans
            
            #dense = dense - mask_neg
            if trans == 'ex-1':
                dense_pos = (K.exp(dense) - 1. + gamma) * mask_pos
                #dense_neg = (K.exp(dense) - 1. - gamma) * mask_neg
            elif trans == 'ex_right':
                dense_pos = (K.exp(dense) - 1. + gamma) * mask_pos
                #dense_neg = (1. - K.exp(-dense) - gamma) * mask_neg
            elif trans == 'ex_left':
                dense_pos = (1. - K.exp(-dense) + gamma) * mask_pos
                #dense_neg = (K.exp(dense) - 1. - gamma) * mask_neg
            else:
                dense_pos = (dense + gamma) * mask_pos
                #dense_neg = (dense - gamma) * mask_neg
            dense_pos = K.transpose(K.transpose(dense_pos) / K.sum(dense_pos, axis=1))
            #dense_neg = K.transpose(K.transpose(dense_neg) / K.sum(dense_neg, axis=1))
            beta = self.beta
            dense = dense_pos# - beta * dense_neg
            

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
