""" 
author:-aam35
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from urllib import request
import numpy as np
import tensorflow as tf
print("Eager execution enabled:", tf.executing_eagerly())
# tf.enable_eager_execution()
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
#from tensorflow.examples.tutorials.mnist import input_data
from read_data import load_fashion_mnist
import time
# import utils
tf.executing_eagerly()
# Define paramaters for the model
learning_rate = 0.001
batch_size = 256
img_shape = (28, 28)
n_epochs = 50
n_train = 60000
n_test = 10000


CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Step 1: Read in data
fmnist_folder = 'data/fashion'
# download_fashion_mnist(fmnist_folder)
#Create dataset load function [Refer fashion mnist github page for util function]
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)
x_train_full, y_train_full_labels = load_fashion_mnist(fmnist_folder, kind='train')
x_test, y_test_labels = load_fashion_mnist(fmnist_folder, kind='t10k')
x_train_full = np.array(x_train_full, np.float32) / 255.0
x_test = np.array(x_test, np.float32) / 255.0

# Step 2: Create datasets and iterator
# create training Dataset and batch it

n_validation = 10000
x_val = x_train_full[-n_validation:]
y_val_labels = y_train_full_labels[-n_validation:]
x_train = x_train_full[:-n_validation]
y_train_labels = y_train_full_labels[:-n_validation]

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_labels))
train_data = train_data.shuffle(buffer_size=60000).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val_labels))
val_data = val_data.batch(batch_size)
# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_labels))
test_data = test_data.batch(batch_size)
#############################
########## TO DO ############
#############################


# # create one iterator and initialize it with different datasets
# iterator = tf.data.Iterator.from_structure(train_data.output_types, 
#                                            train_data.output_shapes)
# img, label = iterator.get_next()

# train_init = iterator.make_initializer(train_data)	# initializer for train_data
# test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
W = tf.Variable(tf.random.normal(shape=[784, 10], stddev=0.01))
b = tf.Variable(tf.zeros([10]), name="bias")
#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def logistic_regression(x):
    return tf.matmul(x, W) + b
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def cross_entropy_loss(y_pred, y_true):
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true_one_hot))
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
def accuracy(y_pred, y_true):
    predicted_class = tf.argmax(y_pred, axis=1)
    true_class = tf.cast(y_true, dtype=tf.int64)
    correct_preds = tf.equal(predicted_class, true_class)
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))

total_start_time = time.time()
#Step 8: train the model for n_epochs times
for i in range(n_epochs):
    start_time = time.time()
    
    # Training loop
    train_loss, train_acc, n_batches = 0, 0, 0
    for batch_x, batch_y in train_data:
        with tf.GradientTape() as tape:
            pred = logistic_regression(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
        
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))
        
        train_loss += loss
        train_acc += accuracy(pred, batch_y)
        n_batches += 1
    
    # Validation loop
    val_acc, val_loss = 0, 0
    n_val_batches = 0
    for batch_x, batch_y in val_data:
        pred = logistic_regression(batch_x)
        val_loss += cross_entropy_loss(pred, batch_y)
        val_acc += accuracy(pred, batch_y)
        n_val_batches += 1
        
    epoch_duration = time.time() - start_time
    print(f"Epoch {i+1:02d}: "
          f"Train Loss = {train_loss/n_batches:.4f}, Train Acc = {train_acc/n_batches:.4f}, "
          f"Val Loss = {val_loss/n_val_batches:.4f}, Val Acc = {val_acc/n_val_batches:.4f}, "
          f"Duration = {epoch_duration:.2f}s")
	################################
	###TO DO#####
	############
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"Total duration: {total_duration:.2f} seconds")
#Step 9: Get the Final test accuracy
test_acc = 0
n_test_batches = 0
for batch_x, batch_y in test_data:
    pred = logistic_regression(batch_x)
    test_acc += accuracy(pred, batch_y)
    n_test_batches += 1
print(f"\n Final Test Accuracy: {test_acc / n_test_batches:.4f}")
#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

#Random Forest Classifier
print("\n Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(x_train, y_train_labels)
rf_predictions = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test_labels, rf_predictions)
print(f" Random Forest Final Accuracy: {rf_accuracy:.4f}")


#SVM Classifier
print("\n Training SVM model")
subset_size = 10000
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(x_train[:subset_size], y_train_labels[:subset_size])
svm_predictions = svm_model.predict(x_test)
svm_accuracy = accuracy_score(y_test_labels, svm_predictions)
print(f" SVM Final Accuracy: {svm_accuracy:.4f}")


def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set 
sample_images, sample_labels = next(iter(test_data))
images = sample_images[:9].numpy()

# Get the true classes for those images.
y = sample_labels[:9].numpy()

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights 

def plot_weights(weights):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####
    w_values = weights.numpy()
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w_values)
    #TO DO## obtains these value from W
    w_max = np.max(w_values)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w_values[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

predictions = logistic_regression(images)
predicted_labels = tf.argmax(predictions, axis=1).numpy()

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(W.numpy().T) 
clusters = kmeans.labels_
print(f"\nCluster results from k-Means (k={n_clusters}):")
for cluster_id in range(n_clusters):
    # Find all classes belonging to the current cluster
    members = [CLASS_NAMES[i] for i, label in enumerate(clusters) if label == cluster_id]
    print(f"  Cluster {cluster_id}: {', '.join(members)}")

plot_images(images=images, y=y, yhat=predicted_labels)


print("\n Visualizing model weights...")
plot_weights(W)