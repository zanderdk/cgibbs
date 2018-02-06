import libcgibbs as gibbs
import math
from typing import List, Dict, Set, Tuple, Callable
import learn.markov as markov
import tensorflow as tf
from datetime import datetime
now = datetime.now()
import numpy as np

def callback(m:markov.Markov, like:float, step:int, writer):
    m.draw("test.png")
    with open('test.png', 'rb') as f:
        io = f.read()
        img_sum = tf.Summary.Image(encoded_image_string=io)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="MC", image=img_sum),
            tf.Summary.Value(tag="loss", simple_value=like)
        ])
        writer.add_summary(summary, step)
        writer.flush()
        print('{} - log-likelihood {:.3f}'.format(step, like))

def learnGibbs(traces: List[List[int]], lab:List[int], beta:float, maxiter:int, burnin:int, lag:int, func:Callable[[markov.Markov, float, int], None]=callback, iter:int=100):

    def call(m, l, s):
        init = m[2].copy()
        L = m[1].copy()
        trans = m[0].copy()
        m = markov.Markov(trans, L, -1, init)
        func(m, l, s)


    m = gibbs.learn(traces, lab, beta, maxiter, burnin, lag, call, iter)
    init = m[2].copy()
    L = m[1].copy()
    trans = m[0].copy()
    m = markov.Markov(trans, L, -1, init)
    return m

def learnGradientDecent(L: List[int], data:List[List[int]], callback):

    lenMax = -1

    for row in data:
        lenMax = max(lenMax, len(row))

    lenMax += 1

    for row in data:
        while len(row) < lenMax:
            row.append(-1)

    a = np.random.rand(len(L), len(L))

    lMax = max(L)+1

    b = [[1 if L[x] == l else 0 for x in range(len(L))] for l in range(lMax)]
    b = np.array(b)

    p = np.random.rand(1, len(L))

    def log_likelihood(A, B, pi, obs):
        at = tf.transpose(A)

        C1 = tf.reshape(pi, (len(L), 1)) * tf.reshape(B[obs[0]], (len(L), 1))

        def forward(i, prev, sumlogc):
            f = (tf.reshape(B[obs[i]], (len(L), 1)) * at) @ prev
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

    A_ = tf.get_variable('a_', trainable=True, initializer=tf.constant(a, dtype=tf.float64))

    B = tf.get_variable('b_', trainable=False, initializer=tf.constant(b, dtype=tf.float64))

    pi_ = tf.get_variable('pi', trainable=True, initializer=tf.constant(
        p, dtype=tf.float64
    ))

    A = tf.nn.softmax(A_, axis=1)
    pi = tf.nn.softmax(pi_)

    obs = tf.placeholder(dtype=np.int32, shape=(len(data),len(data[0])), name='obs')

    res = total_log_likelihood(A, B, pi, obs)

    loss = -res

    step = tf.train.AdamOptimizer(2).minimize(loss)

    with tf.Session().as_default() as sess:
        merged = tf.summary.merge_all()
        logdir = "train/" + now.strftime("%y%m%d-%h%m%s") + "/"
        train_writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        print('optimizing')
        for i in range(10000):
            vals = sess.run({'ll': loss, 'step': step}, feed_dict={obs: data})
            trans = A.eval()
            init = pi.eval()
            mm = markov.Markov(trans, L, -1, init)
            callback(mm, vals['ll'], i, train_writer)