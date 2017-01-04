import tensorflow as tf
from utils import load
from utils import forex
import numpy as np
import math


bk_days = 10
fw_days = 5
feature_days = 10
features = 3


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



def test_tf():
    flat_row = math.ceil(math.ceil(feature_days / 2) / 2)
    flat_col = math.ceil(math.ceil(features / 2) / 2)
    fx_pairs = load.load_fx_pairs(["EURUSD","AUDUSD", "CHFJPY", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY",
                                   "GBPUSD", "USDCAD", "USDCHF", "USDJPY"])
    stack_X = []
    stack_y = []
    eur_X,eur_y = fx_pairs[0].prepare(bk_days,fw_days,feature_days,True)
    test_X = eur_X.reshape((eur_X.shape[0], feature_days,features,1))
    test_y = np.array(eur_y)
    for i in range(1, len(fx_pairs)):
        pair_X, pair_y = fx_pairs[i].prepare(bk_days,fw_days,feature_days,True)
        stack_X.append(pair_X)
        stack_y.extend(pair_y)
    X = np.vstack(tuple(stack_X))
    X = X.reshape((X.shape[0], feature_days, features,1))
    y = np.array(stack_y)
    train_set = forex.Tensor_Set(X,y)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, feature_days,features,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    W_conv1 = weight_variable([2, 2, 1, 32])
    b_conv1 = bias_variable([32])
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    #
    # x_t = x_image.eval(feed_dict={x: mnist.train.next_batch(50)[0]})
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #
    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    #
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    #
    W_fc1 = weight_variable([flat_row*flat_col*64, feature_days*features])
    b_fc1 = bias_variable([feature_days*features])
    #
    h_pool2_flat = tf.reshape(h_pool2, [-1,  flat_row* flat_col * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    W_fc2 = weight_variable([feature_days*features, 2])
    b_fc2 = bias_variable([2])
    #
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(30000):
        batch = train_set.next_batch(1000)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            # saver.save(sess, "Model/test", global_step=i)
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_X, y_: test_y, keep_prob: 1.0}))