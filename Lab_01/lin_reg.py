"""
author:-aam35
"""
import time

import tensorflow as tf
#import tensorflow.contrib.eager as tfe
import numpy as np
import random
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
#tfe.enable_eager_execution()

id = 1
seed_value = int("83121101100") + id
tf.random.set_seed(seed_value)
np.random.seed(seed_value % (2**32))
random.seed(seed_value)


# Create data
NUM_EXAMPLES = 10000
NOISE_LEVEL = 1.0
NOISE_TYPE = 'uniform'
ADD_WEIGHT_NOISE = False
WEIGHT_NOISE_STDDEV = 0.01
ADD_LR_NOISE = False
LR_NOISE_STDDEV = 0.0005

#define inputs and outputs with some noise 
X = tf.random.normal([NUM_EXAMPLES])  #inputs 
noise = tf.random.normal([NUM_EXAMPLES]) #noise 
y = X * 3 + 2 + noise  #true output

# Create variables.
W = tf.Variable(10.0)
b = tf.Variable(12.0)


train_steps = 15000
learning_rate = 0.001
min_learning_rate = 0.00001
patience = 50
improvement_threshold = 0.000001
best_loss = float('inf')
patience_counter = 0

if NOISE_TYPE == 'uniform':
    print(f"Using Uniform data noise with range [{-NOISE_LEVEL}, {NOISE_LEVEL}]")
    noise = tf.random.uniform([NUM_EXAMPLES], minval=-NOISE_LEVEL, maxval=NOISE_LEVEL)
else: # Default to normal
    print(f"Using Normal (Gaussian) data noise with stddev={NOISE_LEVEL}")
    noise = tf.random.normal([NUM_EXAMPLES], stddev=NOISE_LEVEL)

# Define the linear predictor.
def prediction(x):
  return W * x + b

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
  return tf.reduce_mean(tf.square(y - y_predicted))

def L1_loss(y, y_predicted):
  return tf.reduce_mean(tf.abs(y - y_predicted))

def hybrid_loss(y, y_predicted):
  return squared_loss(y, y_predicted) + L1_loss(y, y_predicted)

def huber_loss(y, y_predicted, m=1.0):
  """Huber loss."""
  a = y - y_predicted
  abs_a = tf.abs(a)
  loss = tf.where(abs_a <= m, 0.5 * tf.square(a), m * (abs_a - 0.5 * m))
  return tf.reduce_mean(loss)

start_time = time.time()

for i in range(train_steps):
  with tf.GradientTape() as tape:
    y_predicted = prediction(X)
    loss = squared_loss(y, y_predicted)
  ###TO DO ## Calculate gradients
  gradients = tape.gradient(loss, [W, b])

  if ADD_LR_NOISE:
    noise_val = tf.random.normal(shape=(), stddev=LR_NOISE_STDDEV)
    current_lr = max(min_learning_rate, learning_rate + noise_val.numpy())

  W.assign_sub(learning_rate * gradients[0])
  b.assign_sub(learning_rate * gradients[1])

  if ADD_WEIGHT_NOISE:
    W.assign_add(tf.random.normal(shape=W.shape, stddev=WEIGHT_NOISE_STDDEV))
    b.assign_add(tf.random.normal(shape=b.shape, stddev=WEIGHT_NOISE_STDDEV))

  curr_loss = float(loss.numpy())
  if best_loss - curr_loss > improvement_threshold:
      best_loss = curr_loss
      patience_counter = 0
  else:
      patience_counter += 1
      if patience_counter >= patience and learning_rate > min_learning_rate:
          learning_rate = max(min_learning_rate, learning_rate / 2.0)
          print("Reducing learning rate to {}".format(learning_rate))
          patience_counter = 0

  if i % 100 == 0:
    print("Step: {}, Loss: {}, W: {}, b: {}".format(i, loss, W.numpy(), b.numpy()))

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

plt.plot(X, y, 'bo',label='org')
plt.plot(X, y * W.numpy() + b.numpy(), 'r',
         label="MSE Loss with uniform noise")
plt.legend()
plt.savefig("lin_reg_MSE_0.001_15000_uniform_noises.png", dpi=150)
plt.show
