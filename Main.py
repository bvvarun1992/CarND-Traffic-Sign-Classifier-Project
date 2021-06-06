"""
A pipeline to build a convolutional neural network to detect traffic signs

The model is trained, validated and tested using the dataset obtained from 
German traffic signs dataset and is also tested using five random images 
downloaded from web.

@author: Varun Venkatesh

"""
import pickle
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import csv
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import os, fnmatch
import glob
import matplotlib.image as mpimg
from preprocessing import *
from LeNet import *

######################### Loading the data ##########################################

training_file = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

########################### Exploring the dataset ##################################
# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

# Shape of a traffic sign image
image_shape = X_train[0].shape

# Number of unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

########################### Data exploration visualization ############################ 

# Figure to plot images of all the unique classes 
X_train_unique = []
y_train_unique = []
labels, index = np.unique(y_train, return_index=True)

for i in range(len(index)):
    X_train_unique.append(X_train[index[i]])
    y_train_unique.append(y_train[index[i]])
     
X_train_unique = np.asarray(X_train_unique)
y_train_unique = np.asarray(y_train_unique)

label_name = {}
with open('./signnames.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        label_name[row[0]] = row[1]

fig, ax0 = plt.subplots(15,3, figsize=(10,25))
ax0 = ax0.ravel()
for i in range(45):
    if i < 43:
        ax0[i].imshow(X_train_unique[i])
        ax0[i].set_title('{}'.format(label_name[str(y_train_unique[i])]))
        ax0[i].set_aspect('auto')
    else:
        ax0[i].axis('off')
plt.tight_layout()
plt.show()

# Figure to plot bar graph showing distribution of classes in dataset
label_train, count_train = np.unique(y_train, return_counts=True)
label_valid, count_valid = np.unique(y_valid, return_counts=True)
label_test, count_test = np.unique(y_test, return_counts=True)

fig, (a1,a2,a3) = plt.subplots(1,3, figsize = (15,5))
a1.bar(label_train,count_train)
a1.set_title('Training data')
a2.bar(label_valid,count_valid)
a2.set_title('Validation data')
a3.bar(label_test,count_test)
a3.set_title('Test data')

for ax in fig.get_axes():
    ax.set(xlabel='Lables', ylabel='Count')
    ax.grid(True)
plt.show()

################### Training pipeline ##################################################

# Preprocessing the training data
X_train = preprocessing(X_train)
X_valid = preprocessing(X_valid)

# Shuffling the training data
X_train, y_train = shuffle(X_train, y_train)

# Setting EPOCHS, batch size and drop out probablity
EPOCHS = 80
BATCH_SIZE = 128

# Training the model
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

# Training pipeline
rate = 0.0008

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Evaluation pipeline
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Training session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    valid_accuracy_epoch = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        valid_accuracy_epoch.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}, Validation Accuracy = {:.3f}".format(training_accuracy, validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

# Plotting accuracy
fig, a = plt.subplots()
a.plot(range(EPOCHS), valid_accuracy_epoch)
a.set_xlabel('EPOCHS')
a.set_ylabel('Accuracy')

###################### Testing the model on test set ###################################

X_test = preprocessing(X_test)

with tf.Session() as sess:
    saver.restore(sess, './lenet')
    
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
################# Testing the model on images downloaded from web #################

# Loading images from directory
images = []
for file in glob.glob('./web_images/*.jpg'):
    image = cv2.resize(mpimg.imread(file), dsize=(32,32), interpolation=cv2.INTER_CUBIC)
    images.append(image)
X_test_web = np.asarray(images)

file_names = fnmatch.filter(os.listdir('./web_images/'),'*.jpg')
y_test_web_str = [i.split('.')[0] for i in file_names]
y_test_web = np.asarray(y_test_web_str).astype(np.float32)

# Visualizing the loaded images
fig, ax1 = plt.subplots(1,5, figsize=(15,10))
for i in range(5):
    ax1[i].imshow(X_test_web[i])
    ax1[i].set_title('Image label: {}'.format(y_test_web[i]))

# Preprocessing images
X_test_web_pre = preprocessing(X_test_web)

# Running the prediction using data from trained model
with tf.Session() as sess:
    saver.restore(sess, './lenet')
    prediction = sess.run(tf.argmax(logits,1), feed_dict={x: X_test_web_pre, keep_prob:1.0})
    softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: X_test_web_pre, keep_prob:1.0})
    print('Labels of images are ',prediction)
    
    performance = evaluate(X_test_web_pre, y_test_web)
    print("Performance Accuracy = {:.3f}".format(performance))

# Printing out top five softmax probablities
with tf.Session() as sess:
    top5 = sess.run(tf.nn.top_k(softmax, k=5, sorted=True))
print(top5)

for i in range(len(top5[0])):
    print('Top five probablities for image with label '+y_test_web_str[i]+':')
    for j in range(len(top5[0][i])):
        probablity = top5[0][i][j]
        index = top5[1][i][j]
        print("    Probability = {:.3f}, Label = {:.3f}".format(probablity, index))
        print()
