import libcgibbs as gibbs
import math
from typing import List, Dict, Set, Tuple, Callable
import learn.markov as markov

def learnGibbs(traces: List[List[int]], lab:List[int], beta:float, maxiter:int, burnin:int, lag:int, func:Callable[[markov.Markov, float, int], None], iter:int):

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