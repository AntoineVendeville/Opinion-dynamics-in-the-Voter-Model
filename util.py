# -*- coding: utf-8 -*-
""" Custom functions used in main code """

import numpy as np
from scipy.linalg import expm

# stubborn in one camp only
def section4_simu(n, n1, s1, max_time, spacing, seed=None, warning=False, return_N1=True, return_cvtime=False):
        
    # init
    np.random.seed(seed)
    opinion = np.random.permutation([1]*n1 + [0]*(n-n1))
    stubborn = np.random.choice(np.where(opinion==1)[0], size=s1, replace=False)
    non_stub = np.delete(np.arange(n), stubborn)
    if return_N1:
        N1 = [int(n1)]
    n1_tmp = int(n1)
    
    # iter
    t, t_sum = 0, 0
    while t<max_time and n1_tmp<n:
        if np.random.random() > s1/n:
            u = np.random.choice(non_stub)
            old_opi = opinion[u]
            opinion[u] = np.random.choice(opinion[range(n)])
            n1_tmp += opinion[u] - old_opi
        waiting = np.random.exponential(1/n) # random exp of param n
        t, t_sum = t+waiting, t_sum+waiting
        if t_sum >= spacing and return_N1:
            N1.append(int(n1_tmp))
            t_sum = 0

    # raise warning if not CV
    if warning and n1_tmp<n:
        print("Warning: n1={}, N1t_final={}, n={}".format(n1,n1_tmp,n))
        
    # complete if needed
    if return_N1:
        length = int(np.floor(max_time/spacing)+1)
        if len(N1) < length:
            N1 += [n1_tmp]*(length-len(N1))
    
    # end
    to_return = list()
    if return_N1:
        to_return.append(np.array(N1))
    if return_cvtime:
        to_return.append(t)
    return to_return


# stubborn in both camps
def section5_simu(n, n1, s1, s0, max_time, spacing, seed=None):
        
    # init
    np.random.seed(seed)
    opinion = np.random.permutation([1]*n1 + [0]*(n-n1))
    N1 = [int(n1)]
    n1_tmp = int(n1)
    
    # stubborn
    stub0 = np.random.choice(np.where(opinion==0)[0], size=s0, replace=False)
    stub1 = np.random.choice(np.where(opinion==1)[0], size=s1, replace=False)
    
    # iter
    t, t_sum = 0, 0
    while t<max_time:
        u = np.random.choice(range(n))
        if u not in stub0 and u not in stub1:
            old_opi = opinion[u]
            opinion[u] = np.random.choice(opinion[range(n)])
            n1_tmp += opinion[u] - old_opi
        waiting = np.random.exponential(1/n) # random exp of param n
        t, t_sum = t+waiting, t_sum+waiting
        if t_sum >= spacing:
            N1.append(int(n1_tmp))
            t_sum = 0
            
    # complete if needed
    length = int(np.floor(max_time/spacing)+1)
    if len(N1) < length:
        N1 += [n1_tmp]*(length-len(N1))
    
    # end
    return np.array(N1)


# Q matrix
def Qmatrix(n,s0,s1):
    Q = np.zeros((n-s0-s1+1, n-s0-s1+1))
    for k in range(s1,n-s0+1):
        Q[k-s1,k-s1-1] = (k-s1)*(n-k)/(n-1)
        if k<n-s0:
            Q[k-s1,k-s1+1] = k*(n-k-s0)/(n-1)
        Q[k-s1,k-s1] = - (k-s1)*(n-k)/(n-1) - k*(n-k-s0)/(n-1)
    return Q


# stationary prob
def stationary(n,s0,s1,Q):
    prod = dict()
    pi_s1 = 0 # compute for state s1 first
    for k in range(s1+1, n-s0+1):
        prod[k] = 1
        for i in range(s1,k):
            prod[k] *= Q[i-s1,i-s1+1] / Q[i-s1+1,i-s1]
        pi_s1 += prod[k]
    pi_s1 = 1/(1+pi_s1)
    return np.array([pi_s1] + [pi_s1*prod[k] for k in range(s1+1, n-s0+1)])                   


# mixing time
def mixing_time(n, n1, s0, s1, Q, pi, eps, spacing):
    t = 0
    totalvar = eps+1
    while totalvar > eps:
        Pt = expm(t*Q)
        totalvar = 0.5 * np.abs(Pt[n1-s1,:]-pi).sum()
        t += spacing
    return t


# polarisation
def polarisation(neighbors, clique, N1, S1, S0, max_time, spacing, seed=None):
    
    # init
    np.random.seed(seed)
    n = len(neighbors)
    length = int(np.floor(max_time/spacing)+1)
    
    # opinion
    opinion = np.zeros(n)
    stubborn = set()
    clique_index = np.zeros(n, dtype=int)
    for c,members in enumerate(clique):
        clique_index[members] = c
        opinion[members] = np.random.permutation([1]*N1[c] + [0]*(len(members)-N1[c]))
        stub0 = np.random.choice(np.where(opinion[members]==0)[0]+members[0], size=S0[c], replace=False)
        stub1 = np.random.choice(np.where(opinion[members]==1)[0]+members[0], size=S1[c], replace=False)
        stubborn = stubborn.union(stub0).union(stub1)
    
    # nb opinion-1 count
    N1t = np.zeros((len(clique), length))
    n1_tmp = list()
    for c in range(len(clique)):
        N1t[c][0] = N1[c]
        n1_tmp.append(N1[c])
    n1_index = 1
    
    # iter
    t, t_sum = 0, 0
    while t<max_time and sum(n1_tmp) not in {0,n}:
        u = np.random.choice(range(n))
        if u not in stubborn:
            old_op = opinion[u]
            opinion[u] = np.random.choice(opinion[neighbors[u]])
            n1_tmp[clique_index[u]] += opinion[u] - old_op
        waiting = np.random.exponential(1/n) # random exp of param n
        t, t_sum = t+waiting, t_sum+waiting
        if t_sum >= spacing:
            for c in range(len(clique)):
                N1t[c][n1_index] = n1_tmp[c]
            n1_index += 1
            t_sum = 0
        
    # complete if needed
    if n1_index < length:
        for c in range(len(clique)):
            N1t[c][n1_index:] = np.array([n1_tmp[c]]*(length-n1_index))
    
    # end
    return N1t