from typing import List, Dict, Set, Tuple
import random
import math
import subprocess
random.seed(6)
import numpy as np
import learn.markov as markov
np.random.seed(6)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
from learn.learner import learnGibbs
import tensorflow as tf
from datetime import datetime
now = datetime.now()


def ipv4(n: int, p: float, q: float) -> markov.Markov:
    l = []
    sta1 = np.zeros((n + 3))
    sta2 = np.zeros((n + 3))
    sta3 = np.zeros((n + 3))
    sta1[1] = 1 - q
    sta1[3] = q
    sta2[1] = 1
    sta3[2] = 1
    l.append(sta1)
    l.append(sta2)
    l.append(sta3)
    for x in range(0, n-1):
        ll = np.zeros((n + 3))
        ll[3+1+x] = p
        ll[0] = 1 - p
        l.append(ll)
    l.append(np.zeros((n + 3)))
    l[n + 2][2] = p
    l[n + 2][0] = 1 - p
    l = np.array(l)
    labels = [0, 3, 2]
    for x in range(n):
        labels.append(1)
    return markov.Markov(l, labels, 0)

def callback(m:markov.Markov, like:float, step:int, writer):
    m.draw("test.png")
    with open('test.png', 'rb') as f:
        io = f.read()
        img_sum = tf.Summary.Image(encoded_image_string=io)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="MC", image=img_sum),
            tf.Summary.Value(tag="loss", simple_value=-like)
        ])
        writer.add_summary(summary, step)
        writer.flush()
        print(like)

m = ipv4(5, 0.5, 0.2)
m.draw()
traces = [markov.genNondeterministicTrace(m, 9) for _ in range(0, 10000)]
print(traces)

#print("log likelihood original: " + str(markov.logLikelihood(m, traces)))

def log_likelihood(A, B, pi, obs):
    at = tf.transpose(A)

    C1 = tf.reshape(pi, (3,1)) * tf.reshape(B[obs[0]], (3,1))

    def forward(i, prev, sumlogc):
        f = (tf.reshape(B[obs[i]], (3, 1)) * at) @ prev
        c = 1. / tf.reduce_sum(f)
        return tf.add(i, 1), f * c, tf.add(sumlogc, tf.log(c))

    i = tf.constant(1)
    c = lambda i, prev, sumlogc: tf.greater(obs[i], -1)
    b = lambda i, prev, sumlogc: forward(i, prev, sumlogc)
    r = tf.while_loop(c, b, [i, C1, tf.constant(0., dtype=tf.float64)])

    return -r[2]

def total_log_likelihood(A, B, pi, obs):
    fn = lambda o: log_likelihood(A, B, pi, o)
    pt = tf.map_fn(fn, obs, dtype=tf.float64)
    return tf.reduce_sum(pt)


def likelihood(A, B, pi, obs):
    at = tf.transpose(A)

    C1 = tf.reshape(pi, (3,1)) * tf.reshape(B[obs[0]], (3,1))

    def forward(i, prev):
        return tf.add(i, 1), (tf.reshape(B[obs[i]], (3, 1)) * at) @ prev

    i = tf.constant(1)
    c = lambda i, prev: tf.greater(obs[i], -1)
    b = lambda i, prev: forward(i, prev)
    r = tf.while_loop(c, b, [i, C1])

    return tf.reduce_sum(r[1])


with tf.Session().as_default() as sess:
    merged = tf.summary.merge_all()
    logdir = "train/" + now.strftime("%y%m%d-%h%m%s") + "/"
    train_writer = tf.summary.FileWriter(logdir, sess.graph)

    a = np.array([
        [0.5, 0, 0.5],
        [0.2, 0, 0.8],
        [0.7, 0.3, 0.0]
    ])

    b = np.array([
            [1, 0, 1],
            [0, 1, 0],
    ])

    p = np.array([0.5,0.5,0])

    A = tf.get_variable('a_', trainable=True, initializer=tf.constant(a, dtype=tf.float64))

    B = tf.get_variable('b_', trainable=False, initializer=tf.constant(b, dtype=tf.float64))

    pi = tf.get_variable('pi', trainable=True, initializer=tf.constant(
        p, dtype=tf.float64
    ))

    data = np.array([
        [0,0,0,-1],
        [0,0,-1,-1]
    ])

    obs = tf.placeholder(dtype=np.int32, shape=(len(data),len(data[0])), name='obs')

    res = total_log_likelihood(A, B ,pi ,obs)

    sess.run(tf.global_variables_initializer())

    k = math.exp(sess.run(res, feed_dict={obs: data}))

    print(k)

    #learnGibbs(traces, m.labels, 0.06, 35000, 10000, 10, lambda m,l,s: callback(m,l,s,train_writer), 10)