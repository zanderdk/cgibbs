from typing import List, Dict, Set, Tuple
import random
import math
import subprocess
random.seed(6)

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


def readFile(path: str) -> List[List[int]]:
    res = []
    i = 0
    with open(path) as f:
        for l in f.readlines():
            i += 1
            if i == 1:
                continue
            res.append(l)
    res: List[List[int]] = [[int(y) for y in x.split(' ')] for x in res]
    res: List[List[int]] = [x[1:] for x in res]
    return [x for x in res]

def init_fsm(obs:List[List[int]], A:int, N:int) -> List[Observation]:
    lst: List[Observation] = [] #lable, state

    for l in obs:
        for i,data in enumerate(l):
            state = 0
            if i > 0:
                state = random.randint(0, N)
            lst.append(Observation(data, state))
        lst.append(Observation(A, random.randint(0, N)))
    return lst


def counts_to_wfsa(gibbs_sampled_counts:Dict[Tuple[int, int, int], int], alphabet_size:int, num_states:int, beta:float, ANbeta:float):
    gibbs_counts_sampled_states = [0 for x in range(0, num_states)]
    finalProbs = [0 for x in range(0, num_states)]
    prob = dict()

    for i in range(num_states):
        for j in range(num_states):
            for k in range(alphabet_size):
                gibbs_counts_sampled_states[i] += gibbs_sampled_counts[i,k,j]

    for i in range(num_states):
        for j in range(num_states):
            for k in range(alphabet_size):
                if k == (alphabet_size - 1):
                    finalProbs[i] += (gibbs_sampled_counts[i,k,j] + beta) / (gibbs_counts_sampled_states[i] + ANbeta)
                else:
                    prob[(i,k,j)] = (gibbs_sampled_counts[i,k,j] + beta) / (gibbs_counts_sampled_states[i] + ANbeta)
    return (prob, finalProbs)

def prob_of_data(pfa):
    save_fsa(pfa[0], pfa[1], 'test.fsm')
    args = ("treba --likelihood=f --file test.fsm data.txt").split(" ")
    y = subprocess.check_output(args)
    y = str(y).replace('\'', '').replace('b', '').replace('\\', '').split('n')[0:-1]
    y = [math.log(float(x)) for x in y]
    su = sum(y)
    print(su)

def save_fsa(prob: Dict[Tuple[int, int, int], int], final:List[int], path):
    lst = []
    for tup,pr in prob.items():
        st = f"{tup[0]} {tup[2]} {tup[1]} {pr}\n"
        lst.append(st)
    for i,pr in enumerate(final):
        st = f"{i} {pr}\n"
        lst.append(st)
    with open(path, 'w') as f:
        f.writelines(lst)

def sampler_fsm(chain: List[Observation], beta:float, g_alphabet_size: int, num_states:int, maxiter:int, burnin:int, lag:int):
    alphabet_size = g_alphabet_size + 1
    gibbs_counts: Dict[Tuple[int, int, int], int] = dict()
    gibbs_counts_states: Dict[int, int] = dict()

    gibbs_counts_sampled_states: Dict[int, int] = dict()
    gibbs_sampled_counts: Dict[Tuple[int, int, int], int] = dict()

    for i in range(num_states+1):
        for a in range(alphabet_size):
            for j in range(num_states+1):
                gibbs_counts[(i,a,j)] = 0
                gibbs_sampled_counts[(i,a,j)] = 0
        gibbs_counts_sampled_states[i] = 0
        gibbs_counts_states[i] = 0

    current_prob = [0 for x in range(0, num_states)]

    for i in range(len(chain)-1):
        gibbs_counts_states[chain[i].state] += 1
        gibbs_counts[(chain[i].state, chain[i].sym, chain[i + 1].state)] += 1

    ANbeta = alphabet_size * num_states * beta

    i = 0
    samplecount = 1
    steps = 0
    while (i < maxiter):
        for j in range(0, len(chain) - 1):
            if (j == 0 or (chain[j-1] == g_alphabet_size)):
                continue

            a = chain[j].sym
            aprev = chain[j-1].sym
            z = chain[j].state
            zprev = chain[j-1].state
            znext = chain[j+1].state

            assert (gibbs_counts[(zprev, aprev, z)] > 0)
            assert (gibbs_counts[(z, a, znext)] > 0)

            gibbs_counts[(zprev, aprev, z)] -= 1
            gibbs_counts[(z, a, znext)] -= 1

            gibbs_counts_states[z] -= 1
            g_sum = 0
            for k in range(0, num_states):
                indicator = 1 if (k == zprev and aprev == a and znext == k) else 0
                g_k = (gibbs_counts[(k,a,znext)] + beta + indicator) * (gibbs_counts[(zprev,aprev,k)] + beta) \
                      / (gibbs_counts_states[k] + ANbeta)
                assert(g_k >= 0)
                g_sum += g_k
                current_prob[k] = g_sum

            cointoss = random.random() * g_sum
            low = 0
            high = num_states - 1
            while(low != high):
                mid = int((low + high) / 2)
                if (current_prob[int(mid)] <= cointoss):
                    low = mid + 1
                else:
                    high = mid

            newstate = high

            gibbs_counts[(zprev, aprev, newstate)] += 1
            gibbs_counts[(newstate, a, znext)] += 1

            gibbs_counts_states[newstate] += 1
            chain[j] = Observation(chain[j].sym, newstate)
            steps += 1

        if (i >= burnin and (i - burnin) % lag == 0):
            for l in range(0, len(chain)-1):
                gibbs_sampled_counts[(chain[l].state, chain[l].sym, chain[l + 1].state)] += 1
            samplecount += 1
            pfa = counts_to_wfsa(gibbs_sampled_counts, alphabet_size, num_states, beta, ANbeta)
            prob_of_data(pfa)
        print(i)
        i += 1

states = 20
alphaBet = 4
data = readFile('9.pautomac.train')
data2 = [' '.join([str(y) for y in x] + ['\n']) for x in data]
with open("data.txt", 'w') as f:
    f.writelines(data2)
data = init_fsm(data, alphaBet, states)
sampler_fsm(data, 0.02, alphaBet, states, 100, 2, 2)

