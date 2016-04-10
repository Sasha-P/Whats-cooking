from utils import *
import tensorflow as tf
import numpy as np


cuisine_list, ingredients_list, xs, ys = load_train('vector')
ts, ids = load_test(ingredients_list)

cuisine_count = len(cuisine_list)
ingredients_count = len(ingredients_list)

x = tf.placeholder(tf.float32, [None, ingredients_count])
W = tf.Variable(tf.zeros([ingredients_count, cuisine_count]))
b = tf.Variable(tf.zeros([cuisine_count]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, cuisine_count])

t = tf.placeholder(tf.float32, [None, ingredients_count])

p = tf.nn.softmax(tf.matmul(t, W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

for d in ['/cpu:0', '/gpu:0']:
    with tf.device(d):
        for i in range(100):
            batch_xs, batch_ys = next_batch(xs, ys, 12000)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: xs, y_: ys}))

cls = sess.run(p, feed_dict={t: ts})
save_result('tf_gd', cuisine_list, np.argmax(cls, axis=1).tolist(), ids, 'number')
