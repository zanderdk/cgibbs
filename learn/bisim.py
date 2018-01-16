from learn.markov import Markov
import numpy as np
from pulp import *
from typing import Tuple, Dict, Set
import networkx as nx
import transporter

def minMax(u, v):
    if u < v:
        return u, v
    else:
        return v, u

def transport(v, u, costsMatrix) -> Tuple[np.ndarray, float]:
    prob = LpProblem("transport", LpMinimize)
    x = [[LpVariable("x"+str(i)+"_"+str(j), 0) for j in range(0, len(u))] for i in range(0, len(v))]
    ll = []
    for i,row in enumerate(x):
        for j,x1 in enumerate(row):
            ll.append(x1*costsMatrix[i][j])
    prob += lpSum(ll)

    ll = [[x[i][j] for j, _ in enumerate(row)] for i, row in enumerate(x)]
    for i,l in enumerate(ll):
        prob += lpSum(l) <= v[i]
    ll = [[x[j][i] for j, _ in enumerate(x)] for i, _ in enumerate(x[0])]
    for i,l in enumerate(ll):
        prob += lpSum(l) >= u[i]

    prob.solve()
    k = np.array([[value(x1) for x1 in row] for row in x])
    return k, sum(k)

def solveC(m: Markov, s: int, t:int, d: np.ndarray=None):
    if d is None:
        d = np.ones((len(m.transitions), len(m.transitions)))

    v, u = m.transitions[s], m.transitions[t]

    a = np.zeros((len(v), len(u)))

    mm = transporter.transport(v, u, d, a)

    return a

def solveeC(m: Markov, s: int, t:int, d: np.ndarray=None):
    if d is None:
        d = np.ones((len(m.transitions), len(m.transitions)))

    v, u = m.transitions[s], m.transitions[t]

    vIndex = v.nonzero()[0]
    uIndex = u.nonzero()[0]

    vv = v[vIndex]
    uu = u[uIndex]

    if (len(vv) == 0 or len(uu) == 0):
        return np.zeros(d.shape)

    newd = d[np.ix_(vIndex, uIndex)]

    a = np.zeros((len(vv), len(uu)))

    mm = transporter.transport(vv, uu, newd, a)

    res = np.zeros(d.shape)
    res[np.ix_(vIndex, uIndex)] = a

    return res

def solve(m: Markov, s: int, t:int, d: np.ndarray=None):
    if d is None:
        d = np.ones((len(m.transitions), len(m.transitions)))

    v, u = m.transitions[s], m.transitions[t]

    mm = transport(v, u, d)[0]

    return mm


def solvee(m: Markov, s: int, t:int, d: np.ndarray=None):
    if d is None:
        d = np.ones((len(m.transitions), len(m.transitions)))

    v, u = m.transitions[s], m.transitions[t]

    vIndex = v.nonzero()[0]
    uIndex = u.nonzero()[0]

    vv = v[vIndex]
    uu = u[uIndex]

    if (len(vv) == 0 or len(uu) == 0):
        return np.zeros(d.shape)

    newd = d[np.ix_(vIndex, uIndex)]

    mm = transport(vv, uu, newd)[0]

    res = np.zeros(d.shape)
    res[np.ix_(vIndex, uIndex)] = mm

    return res

def setPair(m: Markov, s:int, t:int, w:np.ndarray, c: Dict[Tuple[int, int], np.ndarray], d: np.ndarray, visited: Set[Tuple[int, int]], exact: Set[Tuple[int, int]]):
    c[(s,t)] = w
    visited.add((s, t))
    for u,v in set(zip(*w.nonzero())):
        u, v = minMax(u, v)
        if (u,v) in visited:
            continue

        visited.add((u,v))
        if u == v:
            d[(u,v)] = 0
            exact.add((u, v))
        elif m.labels[u] != m.labels[v]:
            d[(u,v)] = 1
            exact.add((u, v))

        if (u,v) not in exact:
            w = solveeC(m, u, v, d)
            setPair(m, u, v, w, c, d, visited, exact)

def rg(g, s, t):
    return set(nx.descendants(g, (s,t))) | {(s, t)}

def rc(c, s, t):
    return rg(gc(c), s, t)

def gc(c: Dict[Tuple[int, int], np.ndarray]):
    g = nx.DiGraph()
    edges = [x for x in c.keys()]
    edges = [(x, list(zip(*c[x].nonzero()))) for x in edges]
    edges = [(x, map(lambda tup : minMax(tup[0], tup[1]), y)) for x,y in edges]
    edges = [(x, list(y)) for x,y in edges]
    edges = [[(x, z) for z in y] for x, y in edges]
    edges = sum(edges, [])

    for x in edges:
        g.add_edge(*x)

    return g

def nonzero(c, s, t, exact, d):
    non = set()
    g = gc(c)
    rev = nx.reverse(g)
    for u,v in rc(c, s, t).intersection(exact):
        if d[u][v] <= 0:
            continue
        non.update(rg(rev, u, v))
    return non

def sent(c: Dict[Tuple[int, int], np.ndarray], exact: Set[Tuple[int, int]], s:int, t:int, d: np.ndarray, dic):
    s,t = minMax(s,t)
    lin = list(zip(*c[(s, t)].nonzero()))
    lin = [(minMax(x, y), c[(s, t)][x, y]) for x, y in lin]
    lin = [(p, x) if p not in exact else (d[p[0], p[1]], x) for p, x in lin]
    dic[s,t] = lin

    toComp = {p for p,x in lin if isinstance(p, tuple)}
    for x, y in toComp:
        if (x,y) not in dic:
            sent(c, exact, x, y, d, dic)

def solveDic(dic: Dict, d: np.ndarray, lam: float):
    prob = LpProblem("solve", LpMinimize)
    # dic3 = {}
    # for key,l in dic.items():
    #     kl = [(x, y) if isinstance(x, tuple) else ((-1, -1), x*y) for x,y in l]
    #     dic3[key] = kl
    # print(ddisc.disc(dic3))
    dic2 = dict()
    for x,y in dic.keys():
        dic2[(x, y)] = LpVariable("x"+str(x)+"_"+str(y), 0)

    for x, lin in dic.items():
        l = [(p * c) if not isinstance(p, tuple) else (dic2[p] * c) for p,c in lin]
        prob += dic2[x] == lpSum(l)*lam

    prob.solve()
    for x in prob.variables():
        if x.name != '__dummy':
            na = x.name
            u,v = tuple(na.replace('x', '').split('_'))
            u,v = int(u), int(v)
            d[u,v] = x.value()
    return 0

def disc(c: Dict[Tuple[int, int], np.ndarray], d:np.ndarray, s:int, t:int, exact: Set[Tuple[int, int]], lam:float):
    #non = nonzero(c, s, t, exact, d)
    #for u,v in rc(c, s, t) - non:
    #    d[u][v] = 0
    #    exact.add((u, v))

    dic = dict()
    sent(c, exact, s, t, d, dic)
    solveDic(dic, d, lam)

def aproxMatrix(a,b):
    return np.all(np.isclose(a, b))

def removeEdges(c: Dict[Tuple[int, int], np.ndarray], exact:Set[Tuple[int, int]]):
    for tup in exact:
        if tup in c.keys():
            del c[tup]

def pseudometric(m: Markov, lam: float, q):
    single = False
    if type(q) != set:
        toCompute = {q}
        single = True
    else:
        toCompute = q.copy()
    d = np.ones((len(m.transitions), len(m.transitions)))
    visited = set()
    exact = set()
    c = {}
    while toCompute:
        s,t = toCompute.pop()
        s,t = minMax(s,t)
        if m.labels[s] != m.labels[t]:
            d[s,t] = 1
            exact.add((s,t))
        elif s == t:
            d[s,t] = 0
            exact.add((s,t))
        else:
            if (s,t) not in visited:
                w = solveeC(m, s, t, d)
                setPair(m, s, t, w, c, d, visited, exact)
                disc(c, d, s, t, exact, lam)
            copt = {x:False for x in rc(c, s, t) & c.keys()}
            oldd = d.copy()
            while False in copt.values():
                se = set([tup for tup, flag in copt.items() if not flag])
                u,v = se.pop()
                w = solveeC(m, u, v, d)
                setPair(m, u, v, w, c, d, visited, exact)
                disc(c, d, u, v, exact, lam)
                if not aproxMatrix(d, oldd):
                    copt = {x: False for x in rc(c, s, t) & c.keys()}
                    oldd = d.copy()
                copt[(u,v)] = True
            exact.update(rc(c,s,t))
            removeEdges(c, exact)
            toCompute = toCompute - exact
    d= d * d.T
    if single:
        return d[q[0],q[1]]
    return d