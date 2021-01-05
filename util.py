# -*- coding: utf-8 -*-
""" Custom functions used in main code """

import numpy as np
from scipy.linalg import expm


# simulations
def voter_simu(n, n1, s1, s0, max_time, spacing, seed=None):
        
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