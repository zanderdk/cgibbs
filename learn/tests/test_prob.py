from unittest import TestCase
import learn.markov as markov
import learn.genome as genome
import numpy as np
import math
import prob


def probability(m: markov.Markov, traces):
    res = 0
    for x in traces:
        res += prob.probs(m, x)
    return res

class testProc(TestCase):
    def setUp(self):
        trans = np.array([[0, 1 / 3, 1 / 3, 1 / 6, 0, 1 / 6, 0],
                          [0, 0, 2 / 5, 2 / 5, 0, 1 / 5, 0],
                          [0, 1 / 2, 1 / 2, 0, 0, 0, 0],
                          [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0],
                          [2 / 5, 2 / 5, 0, 0, 0, 0, 1 / 5],
                          [0, 1 / 10, 0, 2 / 5, 1 / 2, 0, 0],
                          [0, 2 / 5, 1 / 3, 0, 1 / 10, 0, 1 / 6]])
        labels = [0, 1, 2, 3, 0, 0, 3]
        m: markov.Markov = markov.Markov(trans, labels, 0)
        self.m = m

    def test_proc(self):
        traces = [[0, 1, 2], [0, 0, 1]]
        realProb = (1/3)*(2/5)*(1/6)*(1/10)
        res = 0
        for x in traces:
            res += prob.probs(self.m, x)
        self.assertAlmostEqual(realProb, math.exp(res))

    def test_procNotAccepting(self):
        tra = np.ones((2, 2)) / 2
        lab = [0, 1]
        m = markov.Markov(tra, lab, 0)
        traces = markov.genTraces(m, 10, 10)
        res = 0
        for x in traces:
            res += prob.probs(self.m, x)
        self.assertEqual(res, float('-inf'))

    def test_basicGenome(self):
        m = genome.basicGenome([0,1,2,3], 0)
        traces = markov.genTraces(self.m, 10, 10)
        res = 0
        for x in traces:
            res += prob.probs(self.m, x)

    def test_nan(self):
        genes = [(0, 3, 1.0, 3, True), (1, 1, 0.3333333333333333, 5, False), (2, 1, 0.5, 9, False), (3, 3, 0.3333333333333333, 15, False), (1, 1, 1.0, 120, True), (2, 2, 1.0, 121, True), (3, 3, 1.0, 122, True), (4, 4, 1.0, 123, True)]
        lables = [0,3,2,3,0]
        posi = {0, 1, 2, 3}
        genes = [genome.Gene(*x) for x in genes]
        g = genome.Genome(genes, lables, posi, 0)
        traces = markov.genTraces(self.m, 10, 10)
        m = g.markov()
        self.assertFalse(math.isnan(probability(m, [traces[0]])))