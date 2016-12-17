import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from data import Data
from data import NEW_SHAPE
import numpy as np
from scipy.misc import imread, imsave, imresize

N = 663

SHAPE = [3, 768, 576]

conv1_filters = 5
conv2_filters = 30
filter_len = 3
CHANNEL = 3
nout = 4
batch_size = 100
total_elements = 60000
learningrate = 1e-3
MAX_STEPS = 1e4
conv1_shape = [filter_len, filter_len, CHANNEL, conv1_filters]
conv2_shape = [filter_len, filter_len, conv1_filters, conv2_filters]
fc_shape = [total_elements, nout]
project_folder = "./"

file_path = [(project_folder + "Crowd_PETS09/View1_merged/frame_%04d.jpg" % i)
              for i in range(N)]
label_path = project_folder + 'Crowd_PETS09/View1_merged/label.csv'

def cross_entropy(pred, y):
    return -tf.reduce_sum(y*tf.log(pred))


def train(dataset):
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, (None, NEW_SHAPE[2], NEW_SHAPE[1], 3), name='image')
        y = tf.placeholder(tf.float32, (None, nout), name='label')
        dropout = tf.placeholder(tf.float32)
        ninput = tf.placeholder(tf.int32)

        out = inference(x, ninput, dropout)

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out, y))

        evaluation = tf.equal(tf.argmax(tf.nn.sigmoid(out), 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(evaluation, tf.float32))

        train_op = tf.train.AdamOptimizer(learningrate).minimize(cost)

        init_op = tf.initialize_all_variables()

        eval_dict = {
            x: dataset.eval_set,
            y: dataset.eval_label,
            ninput: batch_size,
            dropout: 1
        }

        sess = tf.Session()
        sess.run(init_op)

        step = 0

        while step < MAX_STEPS:
            data, label = dataset.next_batch(batch_size)
            feed_dict = {
                x: data,
                y: np.reshape(label, [batch_size, nout]),
                ninput: batch_size,
                dropout: 0.5
            }

            if step % 50 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict=eval_dict)
                tloss, tacc = sess.run([cost, accuracy], feed_dict=feed_dict)
                print('evaluation error = ',loss, ' accuracy = ', acc)

            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)

            step += 1
        loss, acc = sess.run([cost, accuracy], feed_dict=eval_dict)

        print('Finished! evalloss = %.2f' % (loss), "Accuracy= " + "{:.5f}".format(acc))

        sess.close()


def main():
    dataset = Data(file_path, label_path,validation_size=300, ninput=N, nout=nout)
    train(dataset)


def inference(x, ninput, dropout):

    # Helper function begins
    def conv(x, weight, bias):
        conv = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME'), bias)
        return conv

    def max_pool2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def activate(x, a, b):
        return tf.nn.elu(x)

    #Helper function ends

    with tf.name_scope('conv1'):
        weight = tf.get_variable('wc1', shape=conv1_shape,
                                   initializer=xavier_initializer())
        bias = tf.Variable(tf.constant(0, shape=[conv1_filters]))
        conv1 = conv(x, weight, bias)
        conv1 = tf.nn.dropout(conv1, keep_prob=dropout)

    with tf.name_scope('pool1'):
        pool1 = max_pool2(conv1)
        pool1 = activate(pool1, a1, b1)

    with tf.name_scope('conv2'):
        weight = tf.get_variable('wc2', shape=conv2_shape,
                                   initializer=xavier_initializer())
        bias = tf.Variable(tf.constant(0.01, shape=[conv2_filters]))
        conv2 = conv(pool1, weight, bias)
        conv2 = tf.nn.dropout(conv2, keep_prob=dropout)

    with tf.name_scope('pool2'):
        pool2 = max_pool2(conv2)
        pool2 = activate(pool2, a2, b2)

    with tf.name_scope('unroll'):
        #pool1 = max_pool2(pool1)

        unrolled_pool2 = tf.reshape(pool2, [ninput, -1])
        unrolled_pool1 = tf.reshape(pool1, [ninput, -1])
        unrolled = tf.concat(1, (unrolled_pool1, unrolled_pool2))
        unrolled = tf.reshape(unrolled, (ninput, total_elements))

    with tf.name_scope('full'):
        weight = tf.get_variable('wf', shape=fc_shape,
                                  initializer=xavier_initializer())
        bias = tf.Variable(tf.constant(0.01, shape=[nout]))
        fc1 = tf.add(tf.matmul(unrolled, weight), bias)
        out = fc1

    return out


if __name__ == "__main__":
    main()
