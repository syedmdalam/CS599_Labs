# -*- coding: utf-8 -*-
"""CNN_week9.ipynb

IST597 :- Implementing CNN from scratch
Week 9 Tutorial

Author:- aam35
"""

import tensorflow as tf
import numpy as np
import time
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()
# tf.executing_eagerly()
seed = 12345
# tf.random.set_random_seed(seed=seed)
tf.random.set_seed(seed)
np.random.seed(seed)
import matplotlib.pyplot as plt
from read_data import load_fashion_mnist 
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False)

batch_size = 64
hidden_size = 100
learning_rate = 0.01
output_size = 10

class CNN(object):
    def __init__(self, hidden_size, output_size, device=None, norm_type='none'):
        self.device = device
        self.size_output = output_size
        self.norm_type = norm_type
        
        filter_h, filter_w, filter_c, filter_n = 5, 5, 1, 30
        
        # FIXED: changed tf.random_normal -> tf.random.normal
        self.W1 = tf.Variable(tf.random.normal([filter_h, filter_w, filter_c, filter_n], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([filter_n]), dtype=tf.float32)
        
        self.W2 = tf.Variable(tf.random.normal([14*14*filter_n, hidden_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)
        
        self.W3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

        self.variables = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        # Initialize specific parameters based on Norm Type
        if self.norm_type in ['batch', 'layer']:
            self.gamma1 = tf.Variable(tf.ones([filter_n]), dtype=tf.float32)
            self.beta1 = tf.Variable(tf.zeros([filter_n]), dtype=tf.float32)
            self.gamma2 = tf.Variable(tf.ones([hidden_size]), dtype=tf.float32)
            self.beta2 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)
            self.variables.extend([self.gamma1, self.beta1, self.gamma2, self.beta2])

        elif self.norm_type == 'weight':
            W1_val = self.W1.read_value()
            self.V1 = tf.Variable(W1_val)
            norm_w1 = tf.sqrt(tf.reduce_sum(tf.square(W1_val), axis=[0,1,2]))
            self.g1 = tf.Variable(norm_w1)
            W2_val = self.W2.read_value()
            self.V2 = tf.Variable(W2_val)
            norm_w2 = tf.sqrt(tf.reduce_sum(tf.square(W2_val), axis=0))
            self.g2 = tf.Variable(norm_w2)
            self.variables = [self.V1, self.g1, self.b1, self.V2, self.g2, self.b2, self.W3, self.b3]
    
    def flatten(self,X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
        
        X_padded = tf.pad(X, [[0,0], [padding, padding], [padding, padding], [0,0]])

        windows = []
        for y in range(out_h):
            for x in range(out_w):
                window = tf.slice(X_padded, [0, y*stride, x*stride, 0], [-1, window_h, window_w, -1])
                windows.append(window)
        stacked = tf.stack(windows) # shape : [out_h, out_w, n, filter_h, filter_w, c]

        return tf.reshape(stacked, [-1, window_c*window_w*window_h])
    
    def convolution(self, X, W, b, padding, stride):
        n, h, w, c = map(lambda d: d, X.get_shape())
        filter_h, filter_w, filter_c, filter_n = [d for d in W.get_shape()]
        if n is None: n = -1 # Placeholder
            
        out_h = (h + 2*padding - filter_h)//stride + 1
        out_w = (w + 2*padding - filter_w)//stride + 1
        
        X_flat = self.flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
        W_flat = tf.reshape(W, [filter_h*filter_w*filter_c, filter_n])
        z = tf.matmul(X_flat, W_flat) + b
        return tf.transpose(tf.reshape(z, [out_h, out_w, -1, filter_n]), [2, 0, 1, 3])
        
    
        
    def relu(self,X):
        return tf.maximum(X, tf.zeros_like(X))
        
    def max_pool(self,X, pool_h, pool_w, padding, stride):
        n, h, w, c = [d for d in X.get_shape()]
        
        out_h = (h + 2*padding - pool_h)//stride + 1
        out_w = (w + 2*padding - pool_w)//stride + 1

        X_flat = self.flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)

        pool = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h*pool_w, c]), axis=3)
        return tf.transpose(pool, [2, 0, 1, 3])

        
    def affine(self,X, W, b):
        n = tf.shape(X)[0] # number of samples
        X_flat = tf.reshape(X, [n, -1])
        return tf.matmul(X_flat, W) + b 
        
    def softmax(self,X):
        X_centered = X - tf.reduce_max(X) # to avoid overflow
        X_exp = tf.exp(X_centered)
        exp_sum = tf.reduce_sum(X_exp, axis=1)
        return tf.transpose(tf.transpose(X_exp) / exp_sum) 
        
    def batch_normalization(self,x, gamma, beta, epsilon=1e-5):
        mu = tf.reduce_mean(x, axis=0)
        var = tf.reduce_mean((x - mu)**2, axis=0)
        x_hat = (x - mu) / tf.sqrt(var + epsilon)
        z = gamma * x_hat + beta
        return z
    
    def weight_normalization(self,v, g):
        if len(v.shape) == 4: 
             v_norm = tf.sqrt(tf.reduce_sum(v**2, axis=[0,1,2]))
        else: # For Affine
            v_norm = tf.sqrt(tf.reduce_sum(v**2, axis=0))
        w = (g / v_norm) * v
        return w
    
    def layer_normalization(self,x, gamma, beta, epsilon=1e-5):
        mu = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.reduce_mean((x - mu)**2, axis=1, keepdims=True)
        x_hat = (x - mu) / tf.sqrt(var + epsilon)
        z = gamma * x_hat + beta
        return z

    def cross_entropy_error(self,yhat, y):
        return -tf.reduce_mean(tf.log(tf.reduce_sum(yhat * y, axis=1)))
        
    
    def forward(self,X):
        if self.device is not None:
            with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
        
        return self.y
        
        
    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))
        
        
    def backward(self, X_train, y_train):
        """
        backward pass
        """
        # optimizer
        # Test with SGD,Adam, RMSProp
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #predicted = self.forward(X_train)
        #current_loss = self.loss(predicted, y_train)
        #optimizer.minimize(current_loss, self.variables)

        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        #print(predicted)
        #print(current_loss)
        #current_loss_tf = tf.cast(current_loss, dtype=tf.float32)
        grads = tape.gradient(current_loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables))
        return current_loss
    

    def compute_output(self, X):
        if self.norm_type == 'weight':
            W1_curr = self.weight_normalization(self.V1, self.g1)
            W2_curr = self.weight_normalization(self.V2, self.g2)
        else:
            W1_curr = self.W1
            W2_curr = self.W2

        conv_layer1 = self.convolution(X, W1_curr, self.b1, padding=2, stride=1)
        if self.norm_type == 'batch':
            conv_layer1 = self.batch_normalization(conv_layer1, self.gamma1, self.beta1)
        elif self.norm_type == 'layer':
            conv_layer1 = self.layer_normalization(conv_layer1, self.gamma1, self.beta1)
            
        conv_activation = self.relu(conv_layer1)
        conv_pool = self.max_pool(conv_activation, pool_h=2, pool_w=2, padding=0, stride=2)
        conv_affine = self.affine(conv_pool, W2_curr, self.b2)

        if self.norm_type == 'batch':
            conv_affine = self.batch_normalization(conv_affine, self.gamma2, self.beta2)
        elif self.norm_type == 'layer':
            conv_affine = self.layer_normalization(conv_affine, self.gamma2, self.beta2)
            
        conv_affine_activation = self.relu(conv_affine)
        conv_affine_1 = self.affine(conv_affine_activation, self.W3, self.b3)
        return conv_affine_1

def accuracy_function(yhat,true_y):
    correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# Initialize model using GPU
mlp_on_cpu = CNN(hidden_size,output_size, device='gpu')

num_epochs = 10

fmnist_folder = 'data/fashion'
x_train_full, y_train_full_labels = load_fashion_mnist(fmnist_folder, kind='train')
x_test, y_test_labels = load_fashion_mnist(fmnist_folder, kind='t10k')

x_train_full = x_train_full.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_full = tf.one_hot(y_train_full_labels, 10)
y_test = tf.one_hot(y_test_labels, 10)


train_x = tf.convert_to_tensor(x_train_full) # 60,000 samples
train_y = tf.convert_to_tensor(y_train_full)
test_x = tf.convert_to_tensor(x_test)        # 10,000 samples
test_y = tf.convert_to_tensor(y_test)

print(f"Training Shape: {train_x.shape}")
print(f"Testing Shape: {test_x.shape}")
time_start = time.time()
# num_train = 55000
num_train = train_x.shape[0]


mlp_none = CNN(hidden_size, output_size, device='cpu', norm_type='none')

for epoch in range(num_epochs):
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
        .shuffle(buffer_size=1000)\
        .batch(batch_size=batch_size)
        
    loss_total = 0.0
    for inputs, outputs in train_ds:
        loss = mlp_none.backward(inputs, outputs)
        loss_total += loss

    print('Epoch {} - Avg Loss: {:.4f}'.format(epoch + 1, loss_total / (num_train // batch_size)))
    
    # Check Train Accuracy
    preds = mlp_none.compute_output(train_x)
    acc = accuracy_function(preds, train_y) * 100
    print("Training Accuracy = {:.2f}".format(acc.numpy()))

# Check Test Accuracy
preds_test = mlp_none.compute_output(test_x)
acc_test = accuracy_function(preds_test, test_y) * 100
print("Test Accuracy (None) = {:.2f}".format(acc_test.numpy()))

# z= 0
# normalization_types = ['none', 'batch', 'layer', 'weight']
# for norm in normalization_types:
#     print(f"\nTraining with Normalization: {norm}")
#     # Note: Using 'cpu' here to avoid OOM if GPU is small, change to 'gpu' if desired
#     mlp_on_cpu = CNN(hidden_size, output_size, device='cpu', norm_type=norm)
    
#     for epoch in range(num_epochs):
#         train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
#             .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
#             .shuffle(buffer_size=1000)\
#             .batch(batch_size=batch_size)
            
#         loss_total = 0.0
        
#         for inputs, outputs in train_ds:
#             loss = mlp_on_cpu.backward(inputs, outputs)
#             loss_total += loss
        
#         # Calculate average loss per batch for better readability
#         print('Number of Epoch = {} - loss:= {:.4f}'.format(epoch + 1, loss_total / (num_train // batch_size)))
        
#         # Calculate Training Accuracy
#         preds = mlp_on_cpu.compute_output(train_x)
#         accuracy_train = accuracy_function(preds, train_y)
#         accuracy_train = accuracy_train * 100
#         print ("Training Accuracy = {}".format(accuracy_train.numpy()))

#     # Calculate Test Accuracy
#     preds_test = mlp_on_cpu.compute_output(test_x)
#     accuracy_test = accuracy_function(preds_test, test_y)
#     accuracy_test = accuracy_test * 100
#     print ("Test Accuracy = {}".format(accuracy_test.numpy()))

# print(f"\nTotal Time: {time.time() - time_start:.2f}s")
        
# time_taken = time.time() - time_start
# print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
# #For per epoch_time = Total_Time / Number_of_epochs
