# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
from read_data import load_fashion_mnist 

# tf.config.run_functions_eagerly(True) 


num_tasks_to_run = 10           
num_epochs_task1 = 50           
num_epochs_per_task = 20        
batch_size = 256
hidden_units = 256              
n_train = 60000
n_test = 10000
img_shape = (28, 28)
img_size_flat = 784
num_classes = 10


CONFIG_DEPTH = 2               
CONFIG_LOSS = 'nll'             
CONFIG_OPTIMIZER = 'RMSProp'       
CONFIG_DROPOUT_RATE = 0.0      
CONFIG_SEED = 6386058           
CONFIG_LEARNING_RATE = 0.001


np.random.seed(CONFIG_SEED)
tf.random.set_seed(CONFIG_SEED)
random.seed(CONFIG_SEED)

fmnist_folder = 'data/fashion'
x_train_full, y_train_full_labels = load_fashion_mnist(fmnist_folder, kind='train')
x_test, y_test_labels = load_fashion_mnist(fmnist_folder, kind='t10k')
x_train_full = np.array(x_train_full, np.float32) / 255.0
x_test = np.array(x_test, np.float32) / 255.0

n_validation = 10000
x_val = x_train_full[-n_validation:]
y_val_labels = y_train_full_labels[-n_validation:]
x_train = x_train_full[:-n_validation]
y_train_labels = y_train_full_labels[:-n_validation]

train_data_base = tf.data.Dataset.from_tensor_slices((x_train, y_train_labels)).shuffle(n_train - n_validation)
val_data_base = tf.data.Dataset.from_tensor_slices((x_val, y_val_labels))
test_data_base = tf.data.Dataset.from_tensor_slices((x_test, y_test_labels))

task_permutation = []
for task in range(num_tasks_to_run):
  task_permutation.append( tf.constant(np.random.permutation(img_size_flat), dtype=tf.int32) )


def apply_permutation(x, permutation):
    x_flat = tf.reshape(x, [-1, img_size_flat])
    x_permuted = tf.gather(x_flat, permutation, axis=1)
    return x_permuted


class MLP:
    """Manual MLP implementation using tf.Variable."""
    def __init__(self, depth, hidden_units=256, num_classes=10, dropout_rate=0.0):
        self.depth = depth
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.weights = []
        self.biases = []
        self.trainable_variables = []
        
        input_dim = img_size_flat
        for i in range(depth - 1):
            W = tf.Variable(tf.random.normal([input_dim, hidden_units], stddev=0.01), name=f'W_{i}')
            b = tf.Variable(tf.zeros([hidden_units]), name=f'b_{i}')
            self.weights.append(W)
            self.biases.append(b)
            self.trainable_variables.extend([W, b])
            input_dim = hidden_units 

        W_out = tf.Variable(tf.random.normal([input_dim, num_classes], stddev=0.01), name='W_out')
        b_out = tf.Variable(tf.zeros([num_classes]), name='b_out')
        self.weights.append(W_out)
        self.biases.append(b_out)
        self.trainable_variables.extend([W_out, b_out])

    def __call__(self, x_permuted, training=False):
        """Performs the forward pass."""
        x = x_permuted
        for i in range(self.depth - 1):
            x = tf.matmul(x, self.weights[i]) + self.biases[i]
            x = tf.nn.relu(x)
            if self.dropout_rate > 0 and training:
                x = tf.nn.dropout(x, rate=self.dropout_rate)
        logits = tf.matmul(x, self.weights[-1]) + self.biases[-1]
        return tf.nn.softmax(logits)

    def summary(self):
        print(f" Config: Depth={self.depth}, Hidden={self.hidden_units}, Dropout={self.dropout_rate}")
        total_params = 0
        for v in self.trainable_variables:
            total_params += tf.reduce_prod(v.shape)
        print(f"  Total Trainable Params: {total_params.numpy()}")


def nll_loss(y_true, y_pred_probs):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred_probs))


def l1_loss(y_true, y_pred_probs):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    return tf.reduce_mean(tf.abs(y_true_one_hot - y_pred_probs))


def l2_loss(y_true, y_pred_probs): 
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    return tf.reduce_mean(tf.square(y_true_one_hot - y_pred_probs))


def hybrid_loss(y_true, y_pred_probs):
    return l1_loss(y_true, y_pred_probs) + l2_loss(y_true, y_pred_probs)

loss_map = {
    'nll': nll_loss, 'l1': l1_loss, 'l2': l2_loss, 'hybrid': hybrid_loss
}


opt_map = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=CONFIG_LEARNING_RATE),
    'SGD': tf.keras.optimizers.SGD(learning_rate=CONFIG_LEARNING_RATE),
    'RMSProp': tf.keras.optimizers.RMSprop(learning_rate=CONFIG_LEARNING_RATE)
}


def ACC_cal(R):
    T = R.shape[0]
    return np.mean(R[T-1, :])

def BWT_cal(R):
    T = R.shape[0]
    bwt = 0.0
    for i in range(T - 1): 
        bwt += (R[T-1, i] - R[i, i])
    return bwt / (T - 1)

def calculate_accuracy(y_pred_probs, y_true):
    y_pred = tf.argmax(y_pred_probs, axis=1)
    y_true_cast = tf.cast(y_true, tf.int64)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true_cast), tf.float32))
    return correct / tf.cast(tf.shape(y_true)[0], tf.float32)


def train_step(model, x, y, permutation, loss_fn, optimizer):
    x_permuted = apply_permutation(x, permutation)
    
    with tf.GradientTape() as tape:
        y_pred_probs = model(x_permuted, training=True)
        loss = loss_fn(y, y_pred_probs)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    acc = calculate_accuracy(y_pred_probs, y)
    return loss, acc


def eval_step(model, x, y, permutation, loss_fn):
    x_permuted = apply_permutation(x, permutation)
    y_pred_probs = model(x_permuted, training=False) # Dropout is off
    loss = loss_fn(y, y_pred_probs)
    acc = calculate_accuracy(y_pred_probs, y)
    return loss, acc

def evaluate_on_task(model, dataset_base, permutation, loss_fn):
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    
    dataset = dataset_base.batch(batch_size)
    
    for batch_x, batch_y in dataset:
        loss, acc = eval_step(model, batch_x, batch_y, permutation, loss_fn)
        total_loss += loss
        total_acc += acc
        n_batches += 1
        
    return total_loss / n_batches, total_acc / n_batches


print(f"Config: Depth={CONFIG_DEPTH}, Loss={CONFIG_LOSS}, Opt={CONFIG_OPTIMIZER}, Dropout={CONFIG_DROPOUT_RATE}, Seed={CONFIG_SEED}")

R = np.zeros((num_tasks_to_run, num_tasks_to_run))
val_acc_history = {} 

model = MLP(
    depth=CONFIG_DEPTH,
    hidden_units=hidden_units,
    dropout_rate=CONFIG_DROPOUT_RATE
)
model.summary()

loss_fn = loss_map[CONFIG_LOSS]
optimizer = opt_map[CONFIG_OPTIMIZER]

total_start_time = time.time()

for task_id in range(num_tasks_to_run):
    print(f"Training on Task {task_id + 1}/{num_tasks_to_run}")
    print(f" (Permutation {task_id + 1})")
    print("="*10)
    
    permutation = task_permutation[task_id]
    epochs = num_epochs_task1 if task_id == 0 else num_epochs_per_task
    
    task_val_accs = []
    train_data = train_data_base.batch(batch_size)
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        
        # Training
        for batch_x, batch_y in train_data:
            loss, acc = train_step(model, batch_x, batch_y, permutation, loss_fn, optimizer)
            train_loss += loss
            train_acc += acc
            n_batches += 1
            
        val_loss, val_acc = evaluate_on_task(model, val_data_base, permutation, loss_fn)
        task_val_accs.append(val_acc.numpy())
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs}: "
              f"Train Loss = {train_loss/n_batches:.4f}, Train Acc = {train_acc/n_batches:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, "
              f"Duration = {epoch_duration:.2f}s")

    val_acc_history[task_id] = task_val_accs

    print(f"\n Evaluating on tasks 1 to {task_id + 1}")
    for eval_task_id in range(task_id + 1):
        eval_permutation = task_permutation[eval_task_id]
        
        _, test_acc = evaluate_on_task(model, test_data_base, eval_permutation, loss_fn)
        
        R[task_id, eval_task_id] = test_acc
        print(f" Accuracy on Task {eval_task_id + 1}: {test_acc:.4f}")

total_end_time = time.time()
print(f"\nTotal duration: {total_end_time - total_start_time:.2f} seconds")

print("\n" + "="*50)
print("--- Final Results ---")
print("="*50)
print("Final Result Matrix (R):")
print(np.around(R, 4))

acc = ACC_cal(R) 
bwt = BWT_cal(R) 

print(f"\nACC (Average Accuracy): {acc:.4f}")
print(f"BWT (Backward Transfer): {bwt:.4f}")


plt.figure(figsize=(10, 6))
for task_id, accs in val_acc_history.items():
    label = f"Task {task_id+1} ({len(accs)} epochs)"
    plt.plot(accs, label=label, marker='o', markersize=4, linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy During Training for Each Task")
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

plt.figure(figsize=(10, 6))
final_accs = R[num_tasks_to_run - 1, :]
task_labels = [f"Task {i+1}" for i in range(num_tasks_to_run)]
plt.bar(task_labels, final_accs, color='c', edgecolor='k')
plt.xlabel("Task")
plt.ylabel("Final Accuracy")
plt.title(f"Final Model Accuracy on All Tasks (After Training Task {num_tasks_to_run})")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle=':')
for i, acc in enumerate(final_accs):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center')
plt.show()
