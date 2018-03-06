import libcgibbs as gibbs
from typing import List, Callable
import learn.markov as markov
import tensorflow as tf
from datetime import datetime
now = datetime.now()
import numpy as np
import subprocess
from datetime import datetime
now = datetime.now()


def perplexitySol(prts: List[float], c: markov.Markov, traces: List[List[int]]) -> float:
    # su = 0
    # prcs = []
    # for x in traces:
    #     prc = math.exp(prob([x], c))
    #     prcs.append(prc)
    #
    # prcs = np.array(prcs)
    # prcs = prcs / sum(prcs)
    # prts = np.array(prts)
    #
    # for i,_ in enumerate(prts):
    #     pc = prcs[i]
    #     if (pc <= 0):
    #         return float('inf')
    #     su += prts[i] * math.log(pc, 2)
    # return 2 ** (-su)

    prts = np.array(prts)
    prcs = np.array([prob([x], c) for x in traces])

    prcs = np.exp(prcs)

    prcs = prcs / sum(prcs)
    prts = prts / sum(prts)

    prcs = np.log2(prcs)

    su = sum(prts * prcs)
    return 2**(-1.0 * su)



def prob(traces: List[List[int]], m:markov.Markov):
    trans = m.transitions.tolist()
    init = m.initProb
    labels = m.labels

    return gibbs.prob(traces, trans, labels, init)


def learnGibbs(traces: List[List[int]], lab:List[int], beta:float, maxiter:int, burnin:int, lag:int, func:Callable[[markov.Markov, float, int], None]):

    def call(m, l, s):
        init = m[2].copy()
        L = m[1].copy()
        trans = m[0].copy()
        m = markov.Markov(trans, L, -1, init)
        return func(m, l, s)


    m = gibbs.learn(traces.copy(), lab.copy(), beta, maxiter, burnin, lag, call)
    init = m[2].copy()
    L = m[1].copy()
    trans = m[0].copy()
    m = markov.Markov(trans, L, -1, init)
    return m

def learnAlergia(traces: List[List[int]], els: float, callback=None, iter:int = 100, maxIter=35000) -> markov.Markov:
    with open('tra.txt', 'w') as f:
        lst = []
        for y in traces:
            tr = [str(x) for x in y]
            st = ''.join(tr)
            f.write(st)
            f.write('\n')
    args = 'java -jar alergia.jar tra.txt'.split(' ')
    args.append(str(els))
    print(subprocess.check_output(args))
    with open('tra.txt', 'r') as f:
        labels = str(f.readline())
        labels = [int(x) for x in labels.split(' ')]
        l = []
        for x in labels:
            ll = str(f.readline())
            l.append([float(x) for x in ll.split(' ')])
        tra = np.array(l)

    m = markov.Markov(tra, labels, 0)
    like = markov.logLikelihood(m, traces)

    if callback is not None:
        for i in range(0, maxIter, iter):
            callback(m, -like, i)

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