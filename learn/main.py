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

m = ipv4(5, 0.5, 0.2)
m.draw()
traces = [markov.genNondeterministicTrace(m, 9) for _ in range(0, 10000)]
print("log likelihood original: " + str(markov.logLikelihood(m, traces)))

parameters = {
    'data': 'ipv4',
    'beta': 0.06,
    'maxiter': 35000,
    'burnin': 5000,
    'lag': 100,
}

with tf.Session().as_default() as sess:
    merged = tf.summary.merge_all()
    par = ''
    for x,y in parameters.items():
        par += str(x)
        par += ': '
        par += str(y)
        par += ' '
    logdir = "train/" + par + ' at ' + now.strftime("%d/%m/%y-%H:%M") + "/"
    train_writer = tf.summary.FileWriter(logdir, sess.graph)

    learnGibbs(traces, m.labels, parameters['beta'], parameters['maxiter'], parameters['burnin'], parameters['lag'])