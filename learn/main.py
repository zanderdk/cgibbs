from typing import List, Tuple
import random
random.seed(6)
import numpy as np
import learn.markov as markov
np.random.seed(6)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
from learn.learner import learnGibbs, learnAlergia, prob, perplexitySol
import tensorflow as tf
from datetime import datetime
now = datetime.now()
import os
import sys
import requests

draw = False

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

def parStr(parDic):
    par = ''
    for x,y in parDic.items():
        par += str(x)
        par += ': '
        par += str(y)
        par += ' '
    return par

def readTrainSet(path:str) -> Tuple[int, List[List[int]]]:
    x = path
    l = x.split('\n')
    first = l.pop(0)
    _, maxLabel = tuple(first.split(' '))
    maxLabel = int(maxLabel)
    arr: List[List[int]] = []
    for line in l:
        li = line.split(' ')
        li = [int(x) for x in li if x] + [maxLabel,maxLabel]
        if li:
            arr.append(li)
    for row in arr:
        row.pop(0)
    return maxLabel, arr[:-1]


def readSolution(path:str) -> List[float]:
    x = path
    l = x.split('\n')
    l.pop(0)
    arr = []
    for line in l:
        if line:
            li = float(line)
            arr.append(li)
    return arr


problem = int(sys.argv[1])
path = 'http://ai.cs.umbc.edu/icgi2012/challenge/Pautomac/competition/downloads/{}.pautomac.{}'
solutionPath = 'http://ai.cs.umbc.edu/icgi2012/challenge/Pautomac/competition/solutions/{}.pautomac_solution.txt'

train = requests.get(path.format(problem, 'train')).text
test = requests.get(path.format(problem, 'test')).text
solution = requests.get(solutionPath.format(problem)).text

maxLabel, traces = readTrainSet(train)
_, testSet = readTrainSet(test)
sol = readSolution(solution)
traces += testSet

lowest = float('inf')
Glike = 0

def callback(m:markov.Markov, like:float, step:int, writer):
    global Glike, lowest
    #perplexNoPrune = perplexitySol(sol, m, testSet)
    #m.prune()
    testLike = -prob(testSet, m)
    perplex = perplexitySol(sol, m, testSet)
    like = -like
    Glike = like
    if draw:
        name = "test" + now.strftime("%H%M%S")
        markov.removeLonleyNodes(m)
        m.draw(name)
        with open(name + '.png', 'rb') as f:
            io = f.read()
            img_sum = tf.Summary.Image(encoded_image_string=io)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="MC", image=img_sum),
                tf.Summary.Value(tag="log-likelihood", simple_value=like),
                tf.Summary.Value(tag="log-likelihood-test", simple_value=testLike),
                tf.Summary.Value(tag="perplexity", simple_value=perplex)
            ])
            writer.add_summary(summary, step)
            writer.flush()

            print('{} - log-likelihood: {:.3f}, log-likelihood test set: {:.3f}, perplexity: {:.3f}'.format(
                    step, like,
                    testLike,
                    perplex))
        os.remove(name + '.png')
        os.remove(name + '.dot')
        if like < lowest:
            lowest = like
            return True
    else:
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="log-likelihood", simple_value=like),
            tf.Summary.Value(tag="log-likelihood-test", simple_value=testLike),
            tf.Summary.Value(tag="perplexity", simple_value=perplex)
        ])
        writer.add_summary(summary, step)
        writer.flush()
        print('{} - log-likelihood: {:.3f}, log-likelihood test set: {:.3f}, perplexity: {:.3f}'.format(step, like,
                                                                                                        testLike,
                                                                                                        perplex))
        if ((like + 2) < lowest):
            lowest = like
            return True
    return False


gibbsParameters = {
    'data': 'pautomac{}'.format(problem),
    'optimizer': 'cgbbis',
    'beta': 0.5,
    'maxiter': 3000,
    'burnin': 1000,
    'lag': 100,
    'lables': [x for x in range(maxLabel+1)]
}

alergiaParameters = {
    'data': 'pautomac{}'.format(problem),
    'optimizer': 'alergia',
    'epsilon': 0.99
}

it = 1

def train(Param):
    with tf.Session().as_default() as sess:
        global it
        merged = tf.summary.merge_all()
        prtParam = Param.copy()
        del prtParam['lables']
        prtParam['it'] = it
        it += 1
        logdir = "train/" + Param['data'] + "/" + parStr(prtParam) + ' at ' + now.strftime("%d/%m/%y-%H:%M") + "/"
        train_writer = tf.summary.FileWriter(logdir, sess.graph)
        print(Param)
        m = learnGibbs(traces, Param['lables'], Param['beta'], Param['maxiter'], Param['burnin'], Param['lag'], lambda m,l,s: callback(m,l,s,train_writer))
        return lab, Glike, m

best = None
lab = gibbsParameters['lables']

def learn():
    global lab, best
    while(True):
        gibbsParameters['lables'] = lab.copy()
        lab, like, m = train(gibbsParameters.copy())
        if best is None or like < best[1]:
            best = (lab, like, m, m.labelEntropy())
            del best[3][maxLabel]

        ent = best[3]
        if not best[3]:
            lowest = 0
            gibbsParameters['maxiter'] = 20000
            gibbsParameters['lag'] = 100
            gibbsParameters['burnin'] = 10000
            gibbsParameters['lables'] = best[0]
            lab, like, m = train(gibbsParameters.copy())
            return m

        print(ent)
        ma = list(sorted([(y,x) for x,y in ent.items()], reverse=True))[0][1]
        del best[3][ma]
        lab = best[0].copy()
        lab.append(ma)
        lab = list(sorted(lab))
        gibbsParameters['lables'] = lab

    return m

beta = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

def learnAllBeta():
    for x in beta:
        gibbsParameters['beta'] = x
        learn()

learnAllBeta()