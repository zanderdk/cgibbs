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

tran =np.array([
    [0, 0.5, 0.5, 0, 0],
    [0, 0, 2/3, 1/3, 0],
    [0, 0.5, 0.5, 0, 0],
    [0, 0.5, 0, 0, 0.5],
    [0, 0.5, 0, 0, 0.5]
])

lab = [0, 1, 0, 1, 2]

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

m = markov.Markov(tran, lab, -1, [0.8, 0.2, 0, 0, 0])
traces = [markov.genNondeterministicTrace(m, random.randint(3, 10)) for _ in range(1000)]
print(markov.logLikelihood(m, traces))
print(traces)
m.draw()

class Observation(tuple):
    __slots__ = []
    def __new__(cls, sym, state):
        return tuple.__new__(cls, (sym, state))
    @property
    def sym(self):
        return tuple.__getitem__(self, 0)
    @property
    def state(self):
        return tuple.__getitem__(self, 1)

def init_markov(obs:List[List[int]], A:int, N:int, L:List[int] = None) -> Tuple[List[Observation], List[int]]: #(sym, state), labling function
    lst: List[Observation] = []  # lable, state
    if L:
        A = max(L) + 1
        N = len(L)
    if L is None:
        labels = [[x for y in range(N)] for x in range(A)]
        labels = sum(labels, [])
        L = labels
    for x in obs:
        for y in x:
            choises = [i for i, x in enumerate(L) if x == y]
            lst.append(Observation(y, random.choice(choises)))
        lst.append(Observation(A, N))
    return lst, L + [A]

def counts_to_markov(gibbs_sampled_counts:Dict[Tuple[int, int], int], num_states:int, beta:float, ANbeta:float):
    gibbs_counts_sampled_states = [0 for x in range(0, num_states)]
    prob = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            gibbs_counts_sampled_states[i] += gibbs_sampled_counts[i,j]

    for i in range(num_states):
        for j in range(num_states):
            prob[i,j] = (gibbs_sampled_counts[i,j] + beta) / (gibbs_counts_sampled_states[i] + ANbeta)
    return prob

def sampler_markov(chain: List[Observation], L:List[int], beta:float, maxiter:int, burnin:int, lag:int):
    alphabet_size = max(L)
    num_states = len(L)
    gibbs_counts: Dict[Tuple[int, int], int] = dict()
    gibbs_counts_states: Dict[int, int] = dict()

    gibbs_counts_sampled_states: Dict[int, int] = dict()
    gibbs_sampled_counts: Dict[Tuple[int, int], int] = dict()

    for i in range(num_states):
        for j in range(num_states):
            gibbs_counts[(i,j)] = 0
            gibbs_sampled_counts[(i,j)] = 0
        gibbs_counts_sampled_states[i] = 0
        gibbs_counts_states[i] = 0

    current_prob = [0 for x in range(0, num_states)]

    for i in range(len(chain)-1):
        gibbs_counts_states[chain[i].state] += 1
        gibbs_counts[(chain[i].state, chain[i + 1].state)] += 1

    ANbeta = alphabet_size * num_states * beta

    i = 0
    samplecount = 1
    steps = 0
    while (i < maxiter):
        for j in range(0, len(chain) - 1):
            if (j == 0 or chain[j].sym == alphabet_size):
                continue

            a = chain[j].sym
            z = chain[j].state
            zprev = chain[j-1].state
            znext = chain[j+1].state

            assert (gibbs_counts[(zprev, z)] > 0)
            assert (gibbs_counts[(z, znext)] > 0)

            gibbs_counts[(zprev, z)] -= 1
            gibbs_counts[(z, znext)] -= 1

            gibbs_counts_states[z] -= 1

            g_sum = 0
            for k in range(0, num_states):
                indicator = 1 if (k == zprev and znext == k) else 0
                g_k = ((gibbs_counts[(k, znext)] + beta + indicator) * (gibbs_counts[(zprev, k)] + beta)
                      / (gibbs_counts_states[k] + ANbeta)) * 1 if L[k] == a else 0
                assert (g_k >= 0)
                g_sum += g_k
                current_prob[k] = g_k

            current_prob = [x/g_sum for x in current_prob]

            newstate = np.random.choice(range(0, len(current_prob)), 1, p=current_prob)[0]

            assert (L[newstate] == a)

            gibbs_counts[(zprev, newstate)] += 1
            gibbs_counts[(newstate, znext)] += 1

            gibbs_counts_states[newstate] += 1
            chain[j] = Observation(chain[j].sym, newstate)
            steps += 1

        if (i >= burnin and (i - burnin) % lag == 0):
            for l in range(0, len(chain)-1):
                gibbs_sampled_counts[(chain[l].state, chain[l + 1].state)] += 1
            samplecount += 1
            m = counts_to_markov(gibbs_sampled_counts, num_states, beta, ANbeta)
            initProb = m[-1,:-1]
            s = sum(initProb)
            for g,x in enumerate(initProb):
                initProb[g] = x/s
            m = m[:-1,:-1]
            for row in m:
                s = sum(row)
                for g,x in enumerate(row):
                    row[g] = x/s
            m = markov.Markov(m, L[:-1], -1, initProb)
            m.draw("data.png")
            print(markov.logLikelihood(m, traces))
            print(m.initProb)

        print(i)
        i += 1

L = m.labels
mm = markov.alergia(traces, 0.95)
#print("logLiklihood alergia: " + str(markov.logLikelihood(mm, traces)))
path, L = init_markov(traces, 4, 5, L)
print(path)
print(L)
sampler_markov(path, L, 0.5, 10000, 10, 10)
