import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

'''First CNN model'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('D:\Python\Lib\site-packages\keras\datasets', reshape = False)
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels

# Pad images with 0s
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

index = random.randint(0, len(x_train))
image = x_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

x = tf.placeholder(tf.float32, (None, 32, 32, 1))  # X: [batch size, features]
y = tf.placeholder(tf.int32, (None))  # output size: [batch size, output classes]
one_hot_y = tf.one_hot(y, 10)

def LeNet(x):
    # define hypterparameters
    mu = 0
    sigma = 0.1
    sess2 = tf.Session()

    # Layer 1: Convolution
    c1_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 6], mean = mu, stddev = sigma)) # define random initial values by normal distirbution, extract 32 feature maps from input image
    c1_b = tf.Variable(tf.zeros(6))
    c1_out = tf.nn.conv2d(x, c1_w, strides = [1,1,1,1], padding = 'VALID') + c1_b
    c1_out = tf.nn.relu(c1_out)

    # Layer 2:Pooling
    s2 = tf.nn.max_pool(c1_out,
                        ksize = [1,2,2,1], # pooling in 4 dimensions, we don't want to do pooling on batch & channel dimension so set to 1
                        strides = [1,2,2,1], # same as ksize, the sliding steps of pooling block on 4 dimension
                        padding = 'VALID' # if padding = VALID, then when sliding windows exceed image dimension, output will abandoned the pooling at that step. If padding = SAME, it will pad 0 to new axis to finished the last pooling.
                        )

    # Layer 3: Convolution
    c3_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 6, 16], mean = mu, stddev = sigma)) # Extract 64 feature maps in C3 layer
    c3_b = tf.Variable(tf.zeros(16))
    c3_out = tf.nn.conv2d(s2, c3_w, strides=[1,1,1,1], padding = 'VALID') + c3_b
    c3_out = tf.nn.relu(c3_out)

    # Layer 4: Pooling
    s4 = tf.nn.max_pool(c3_out, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # Layer 5: Convolution layer 5
    f5_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 16, 120], mean = mu, stddev = sigma))
    f5_b = tf.Variable(tf.zeros(120))
    f5_out = tf.nn.conv2d(s4, f5_w, strides = [1,1,1,1], padding = 'VALID') + f5_b
    f5_out = tf.squeeze(tf.nn.relu(f5_out))  # remove one dimension entry, size = {samples, 120}

    # Layer 6: Full connected layer 6
    f6_w = tf.Variable(tf.truncated_normal(shape = [120, 84], mean = mu, stddev = sigma))
    f6_b = tf.Variable(tf.zeros(84))
    f6_out = tf.matmul(f5_out, f6_w) + f6_b

    # Layer 7: Full connected layer 7
    f7_w = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=mu, stddev=sigma))
    f7_b = tf.Variable(tf.zeros(10))
    logit = tf.matmul(f6_out, f7_w) + f7_b
    return logit


predict_y = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = predict_y, labels = one_hot_y) # labels should be one hot coding
loss = tf.reduce_mean(cross_entropy)
Optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(one_hot_y, 1))  # tf.argmax() converts labels back to real value classes, tf.equal() return True if sample of 2 matrix is equal
# correct_prediction = [True, False, False, True, True, ....]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''Mini-batch learning'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch = 10
    batch_size = 128
    train_samples = len(x_train)
    print('Training samples:', x_train.shape)

    for iterate in range(epoch):
        x_train, y_train = shuffle(x_train, y_train)
        # train for each batch
        print('Training epoch: {}'.format(iterate + 1))
        for batch_step in range(0, train_samples, batch_size):
            end = batch_step + batch_size
            batch_x, batch_y = x_train[batch_step:end], y_train[batch_step:end]
            sess.run(Optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test})

        print('Testing Accuracy: ', acc)

    plt.show()






