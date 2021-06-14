# -*- coding: utf-8 -*-
""" Custom functions used in main code """

import numpy as np
import networkx as nx
from scipy.linalg import expm, null_space

# voter model with zealots in both camps
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



# stubborn in both camps with custom graph
def custom_graph_simu(leaders, n1, s1, s0, max_time, spacing, seed=None):
    """leaders must be a dict with numpy arrays as values"""
        
    # init
    n = len(leaders)
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
        lead = leaders[u]
        if u not in stub0 and u not in stub1 and lead.size>0: # skip if stubborn or no leaders
            old_opi = opinion[u]
            opinion[u] = np.random.choice(opinion[lead])
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



# custom function to create a follwoers dict for an ER graph of cosen parameter, directed or not
# /!\ RETURNS LEADERS DICT (and not followers)
def create_ER_graph(n,p, directed=False):
    leaders = {i:np.array([], dtype=int) for i in range(n)}
    
    if directed:
        for i in range(n):
            for j in range(i):
                if np.random.random() < p:
                    leaders[i] = np.append(leaders[i],j)
                if np.random.random() < p:
                    leaders[j] = np.append(leaders[j],i)
                
    else:
        for i in range(n):
            for j in range(i):
                if np.random.random() < p:
                    leaders[i] = np.append(leaders[i],j)
                    leaders[j] = np.append(leaders[j],i)
        
    return leaders

# discrete chain transitions
def Pmatrix(n,z0,z1):
    P = np.zeros((n-z0-z1+1, n-z0-z1+1))
    for k in range(z1,n-z0+1):
        P[k-z1,k-z1-1] = (k-z1)/n * (n-k)/(n-1)
        if k<n-z0:
            P[k-z1,k-z1+1] = (n-k-z0)/n * k/(n-1)
            P[k-z1,k-z1] = 1 - P[k-z1,k-z1-1] - P[k-z1,k-z1+1]
        else:
            P[k-z1,k-z1] = 1 - P[k-z1,k-z1-1]
    return P


# discrete chain stationary
def stationary_discrete(P):
    pi = null_space(P.T-np.identity(P.shape[0]))
    pi = pi/pi.sum() # normalise
    pi = np.squeeze(pi) # flatten
    return pi


# distribution of the chain at step t according to spectral representation, can choose discrete or continuous chain
def distrib_spectral(n,z0,z1,n1,t,mode="continuous"):
    
    # get S matrix and its spectrum
    P = Pmatrix(n,z0,z1)
    pi_sqrt = np.sqrt(stationary_discrete(P))
    S = np.diag(pi_sqrt).dot(P).dot(np.diag(1/pi_sqrt))
    L, U = np.linalg.eig(S) # eigval, eigvec
    
    # sort eigvals and eigvecs
    idx = L.argsort()[::-1]
    L = L[idx]
    U = U[:,idx]
    
    # compute stationary 
    res = list()
    if mode == "continuous":
        for k in range(z1,n-z0+1):
            val = pi_sqrt[k-z1]/pi_sqrt[n1-z1] * (np.exp(L-1)*t * U[:,n1-z1] * U[:,k-z1]).sum()
            res.append(val)
    elif mode == "discrete":
        for k in range(z1,n-z0+1):
            val = pi_sqrt[k-z1]/pi_sqrt[n1-z1] * (L**t * U[:,n1-z1] * U[:,k-z1]).sum()
            res.append(val)
    else:
        print("mode should be continuous or discrete")
        
    # end
    return np.array(res)



######################## OPTIM PROBLEM ###########################
# ideal z1 (no backfire)
def z1_ideal(z0,lambd):
    return int(z0*lambd/(1-lambd))

# ideal z1 (with backfire)
def z1_ideal_backfire(z0,lambd,alpha):
    D = 1-lambd-alpha*lambd*z0
    if D>0:
        return int(lambd*z0/D)
    else:
        return None

# cv time of coalescing random walks on complete graph with n nodes and z total zealots
def coalesce_cvtime(n,z):
    res = 0
    for k in range(1,n-z+1):
        res += 1/(k*(k+z-1))
    res *= n-1
    return res

# create a leaders dict for a specified graph model
def create_connected_user_graph(n,model,param): #param should be w for ER, m for BA, (k_ws,p) for WS, lambda for SF
    connected = False # restart if we don't get a connected graph
    while not connected:
        if model=="ER":
            G = nx.erdos_renyi_graph(n,param)
        elif model == "BA":
            G = nx.barabasi_albert_graph(n,param)
        elif model == "WS":
            G = nx.watts_strogatz_graph(n,4,param) # initially 4 neighborus for each node
        elif model == "SF":
            even = False
            while not even: # retstart as long as sum of degrees is not even
                deg_seq = np.random.zipf(param,n)
                even = deg_seq.sum()%2==0
            G = nx.configuration_model(deg_seq)
            G = nx.Graph(G) # remove multi-edges
            G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
        else: 
            print("incompatible graph model: {}".format(model))
            return None
        connected = nx.is_connected(G)
    return {i: np.array(list(G.neighbors(i))) for i in range(n)} # return a leader dict

# bound on cv time with n nodes and z total zealots
def bound_cvtime(n,z):
    return np.exp(1)*(1+np.log(n-z))*(n-1)/(z-1)

# greedy algo without backfire effect
def greedy_optim(n,z0,lambd,zmax,T): #cv=cv function
    # init
    z1 = min(zmax, z1_ideal(z0,lambd))
    t = coalesce_cvtime(n,z0+z1)
    # iter
    n_iter = 0
    while t>T or z1>zmax:
        if t>T and z1>=zmax:
            return -1, -1, n_iter
        elif t>T and z1<zmax:
            z1 += 1
        elif t<T and z1>zmax:
            z1 -= 1
        else: # for debug
            return "bug"
        t = coalesce_cvtime(n,z0+z1)
        n_iter += 1 
    # end
    return z1, t, n_iter

# greedy algo with backfire effect and converting method
def greedy_optim_backfire(n,z0,lambd,alpha,zmax,T):
    n_iter = 0
    z1_ideal_tmp = z1_ideal_backfire(z0,lambd,alpha)
    if z1_ideal_tmp==None:
        return -1, -1, n_iter
    else:
        z1 = min(zmax, z1_ideal_tmp)
        t = coalesce_cvtime(n,z0+z1)
        while t>T or z1>zmax:
            if t>T and z1>=zmax:
                return -1, -1, n_iter
            elif t>T and z1<zmax:
                z1 += 1
            elif t<T and z1>zmax:
                z1 -= 1
            else: # for debug
                return "bug"
            t = coalesce_cvtime(n,z0+z1)
            n_iter += 1 
        # end
        return z1, t, n_iter