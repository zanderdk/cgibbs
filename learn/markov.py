from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import randint
import subprocess
import os
import math
import prob
from functools import reduce
import operator
import random

class Node:
    def __init__(self, id:int, child: Dict['int', Tuple[float, 'Node']], label:int = 0):
        self.label = label
        self.childs: Dict['int', Tuple[float, 'Node']] = child
        self.id = id

    def succ(self, lam: int) -> 'Node':
        if lam not in self.childs:
            return 0
        return self.childs[lam][1]

    def succ(self) -> List['Node']:
        return [x[1] for x in self.childs]

    def __eq__(self, other: 'Node'):
        return self.id == other.id

    def __ne__(self, other: 'Node'):
        return not self == other


class Markov:
    def __init__(self, trans: np.ndarray, l: List[int], init: int = -1, initProb:Optional[List[int]] = None):
        self.labels: List[int] = l
        self.transitions: np.ndarray = np.array(trans)
        self.init: int = init
        if initProb is None:
            initProb = [0 for _ in range(len(self.transitions))]
            initProb[init] = 1
        self.initProb = initProb

        assert len(self.transitions) == len(self.labels)

    def graph(self):
        g = nx.DiGraph()
        for i,x in enumerate(self.labels):
            g.add_node(i)

        for i,_ in enumerate(self.transitions):
            for j,_ in enumerate(self.transitions):
                if self.transitions[i,j] > 0:
                    g.add_edge(i, j, weight=self.transitions[i,j])

        return g

    def draw(self, name="graph.png"):
        ma = {
            0:"gray",
            1:"red",
            2:"cyan",
            3:"yellow",
            4:"green"
        }
        l = []
        ll = []
        for id in range(len(self.labels)):
            l.append((id, self.labels[id]))
        for id,lab in l:
            x = (id+1, {'color': ma[lab], 'peripheries': '1', 'style':'filled'})
            ll.append(x)
        ll.append((0, {'color': 'black', 'peripheries': '2', 'style':'filled'}))


        edges = list(zip(*self.transitions.nonzero()))
        edges = [(s+1,t+1,self.transitions[s,t]) for s,t in edges]
        for j,w in enumerate(self.initProb):
            edges.append((0,j+1,w))
        edges = [(str(s),str(t),'%.2f' % w,'black') for s,t,w in edges if w >= 0.01]
        st = "digraph \"\" {\n"
        for id, x in ll:
            s = str(id)
            s += '['
            s += ", ".join([key+"="+val for key,val in x.items()])
            s += '];'
            st += s
            st += '\n'
        st += '\n'
        for s,t,w,c in edges:
            ste = f"{s} -> {t} [label={w}, color={c}];\n"
            st += ste
        st += '}'
        #os.remove("graph.dot")
        with open("graph.dot", 'w') as f:
            f.write(st)
        args = ("sh dot.sh " + name).split(" ")
        subprocess.check_output(args)

    def reachable(self, node:int) -> Set[int]:
        g = self.graph()
        rc: Set[int] = nx.descendants(g, node) | {node}
        return rc

    def stateIds(self) -> Set[int]:
        return set(range(len(self.labels)))


def pathsFromTraces(m: Markov, traces:List[int]) -> List[List[int]]:
    tra = traces.copy()
    paths = [[i] for i,x in enumerate(m.initProb) if x > 0 and tra[0] == m.labels[i]]
    tra = tra[1:]
    while len(tra) > 0:
        for p in paths.copy():
            s = p[-1]
            a = tra[0]
            posibleStates = m.transitions[s].nonzero()[0]
            newPaths = [p + [x] for x in posibleStates if m.labels[x] == a]
            paths.remove(p)
            paths = paths + newPaths
        tra = tra[1:]
    return paths

def pathProb(m: Markov, tra: List[int]) -> float:
    s = tra[0]
    pro = math.log(m.initProb[s])
    for x in tra[1:]:
        pro += math.log(m.transitions[s][x])
        s = x
    return pro

def traceProb(m: Markov, trace: List[int]) -> float:
    probs = [pathProb(m, x) for x in pathsFromTraces(m, trace)]
    pr = probs[0]
    for p in probs[1:]:
        pr = np.logaddexp(pr, p)
    return pr

def logLikelihood(m: Markov, traces: List[List[int]]) -> float:
    probs = [traceProb(m, x) for x in traces]
    return sum(probs)

def genNondeterministicTrace(m: Markov, le = 1000) -> List[int]:
    s = np.random.choice(range(0, len(m.labels)), p=m.initProb)
    trace = [m.labels[s]]
    states = [x for x in range(len(m.labels))]
    for x in range(le-1):
        s = random.choices(states, k=1, weights=m.transitions[s])[0]
        trace += [m.labels[s]]
    return trace

def removeLonleyNodes(m: Markov) -> Markov:
    toRemove = list(m.stateIds() - m.reachable(m.init))
    tra = m.transitions
    tra = np.delete(tra, toRemove, 0)
    tra = np.delete(tra, toRemove, 1)
    lab = np.array(m.labels.copy())
    lab = np.delete(lab, toRemove, 0)
    return Markov(tra, list(lab.tolist()), m.init)

def isMakrovDeterministic(m: Markov) -> bool:
    res = True
    for row in m.transitions:
        ind = row.nonzero()[0]
        la = [m.labels[x] for x in ind]
        res = res and len(la) == len(set(la))
    return res

def joinMarkov(m1: Markov, m2: Markov) -> Tuple[Markov, int]:
    le = len(m1.transitions) + len(m2.transitions)
    len1 = len(m1.transitions)
    len2 = len(m2.transitions)
    m = np.zeros((le, le))
    m[0:len1,0:len1] += m1.transitions
    m[len1:len1+len2,len1:len1+len2] += m2.transitions
    l = m1.labels.copy()
    for x in m2.labels:
        l.append(x)
    return (Markov(m, l, m1.init), len1)

def normalizeFPTA(node: Node):
    s = sum([x[0] for x in node.childs.values()])
    for x in node.childs.keys():
        f,n = node.childs[x]
        f = f / s
        node.childs[x] = (f,n)

    for f,n in node.childs.values():
        normalizeFPTA(n)

def FPTAlabels(node: Node, lst: List[int]):
    l = (node.id, node.label)
    lst.append(l)
    for f,n in node.childs.values():
        FPTAlabels(n, lst)
    return lst

def makrovAddSelfLoop(m: Markov) -> Markov:
    for i,row in enumerate(m.transitions):
        x = 1 - np.sum(row)
        if x > 0:
            row[i] = x
    return m


def FPTAtoMarkov(node: Node):
    ma = maxFTPAId(node)
    matrix = [[0.0 for _ in range(0, ma+1)] for _ in range(0, ma+1)]
    trans = np.array(createMatrix(node, matrix))
    labels = FPTAlabels(node, [])
    labels = sorted(labels, key=lambda x: x[0])
    labels = [x[1] for x in labels]
    return Markov(trans, labels, 0)



def createMatrix(node:Node, m:List[List[float]]):
    ed = [(node.id, n.id, f) for f,n in node.childs.values()]
    for fr,to,w in ed:
        m[fr][to] = w
    for _, n in node.childs.values():
        createMatrix(n, m)
    return m


def maxFTPAId(node: Node, id:int = 0):
    id = max(id, node.id)
    l = [maxFTPAId(n,id) for _,n in node.childs.values()]
    l.append(id)
    return max(l)


def createFPTA(traces: List[List[int]]):
    id = 0
    start = Node(-1, dict(), -1)
    for l in traces:
        current = start
        for x in l:
            if x in current.childs:
                f,n = current.childs[x]
                f += 1
                current.childs[x] = (f,n)
            else:
                current.childs[x] = (1, Node(id, dict(), x))
                id += 1

            _, current = current.childs[x]

    normalizeFPTA(start.childs[0][1])
    return start.childs[0][1]

def genRandomTrace(m: Markov, initState:int = 0, maxLe = 100, n = 100) -> List[List[int]]:
    res = []
    for x in range(0, n):
        res.append(pathToTrace(m, path(m, initState, randint(1, maxLe))))
    return res

def genTraces(m: Markov, le = 100, n = 100) -> List[List[int]]:
    res = []
    for x in range(0, n):
        res.append(pathToTrace(m, path(m, m.init, le)))
    return res


def pathToTrace(m: Markov, pa:List[int]) -> List[int]:
    return [m.labels[x] for x in pa]

def path(m: Markov, initState:int = 0, k:int = 100) -> List[int]:
    return _path(m, [initState], initState, k)

def _path(m: Markov, seq:List[int], initState:int = 0, k:int = 100) -> List[int]:
    states = list(range(0, len(m.transitions)))
    prop = m.transitions[initState]

    if k == 0:
        return seq

    x = np.random.choice(
        states,
        1,
        p=prop
    )[0]

    seq.append(x)
    return _path(m, seq, x, k-1)

def alergia(traces: List[List[int]], els: float) -> Markov:
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
    os.remove('tra.txt')
    return Markov(tra, labels, 0)

def perplexity(t: Markov, c: Markov, traces: List[List[int]]) -> float:
    su = 0
    prts = []
    prcs = []
    for x in traces:
        prt = math.exp(prob.probs(t, x))
        prc = math.exp(prob.probs(c, x))
        prts.append(prt)
        prcs.append(prc)

    prts = np.array(prts)
    prcs = np.array(prcs)
    prts = prts / sum(prts)
    prcs = prcs / sum(prcs)

    for i,_ in enumerate(prts):
        pc = prcs[i]
        if (pc <= 0):
            return float('inf')
        su += prts[i] * math.log(pc)
    return 2 ** (-su)