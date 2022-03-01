import sys
import numpy as np
import gurobi as gp

def get_xbars_complete(z0,z1,a=0):
    return z1/(z0+z1+a*z1)

def get_sigma_complete(z0,z1,a=0):
    return 4*(z0+a*z1)*z1/(z0+z1+a*z1)**2
    
def get_rho_complete(N,z0,z1,a=0):
    z0 = z0+a*z1
    z = z0+z1
    return 2*z0*z1*(N-z) / (z*(z+1)*(N-1))

def max_sigma_complete(N,z0,a=0):
    z1max = (N-z0)/(1+a)
    z1 = min(z1max,z0/(1-a))
    sigma =  get_sigma_complete(z0,z1,a)
    return z1,sigma
    
def get_z0z1(N,W,nz0,nz1):
    z = nz0+nz1
    free = N-z
    z0 = W[:free,free:N-nz1].sum(axis=1)
    z1 = W[:free,N-nz1:N].sum(axis=1)
    return z0,z1

def get_L(N,z,W):
    free = N-z
    return np.diag(W[:free,:free].sum(axis=1))-W[:free,:free]


def get_xstar(N,W,z,z0,z1,L=False):
    if L: # can input  L instead of W
        return np.linalg.inv(W+np.diag(z0)+np.diag(z1)).dot(z1)
    else:
        L = get_L(N,z,W)
        return np.linalg.inv(L+np.diag(z0)+np.diag(z1)).dot(z1)


def get_dxstar(N,a,xstar,L,z0,z1):
    LZ_inv = np.linalg.inv(L+np.diag(z0)+np.diag(z1))
    return (1-a) * (2*xstar-1) * LZ_inv.dot(np.eye(N)-x_star).sum(axis=0) /N


def get_coeffs_complete(N,z0,a):
    return [-2*a**4 - 6*a**3 - 6*a**2 - 2*a,
            -8*a**3*z0 - 4*a**3 - 16*a**2*z0 - 8*a**2 - 8*a*z0 - 4*a,
            2*N*a**2*z0 + 2*N*a**2 + 2*N*a - 2*N*z0 - 12*a**2*z0**2 - 10*a**2*z0 - 14*a*z0**2 - 12*a*z0 - 2*z0**2 - 2*z0,
            4*N*a*z0**2 + 4*N*a*z0 - 8*a*z0**3 - 8*a*z0**2 - 4*z0**3 - 4*z0**2,
            2*N*z0**3 + 2*N*z0**2 - 2*z0**4 - 2*z0**3]


def max_rho_complete(N,z0,a):
    roots = np.roots(get_coeffs_complete(N,z0,a))
    z1max = (N-z0)/(1+a)
    z1min = 0
    rho_max, z1 = -1, -1
    for r in list(roots)+[z1min,z1max]:
        #print('ok1')
        if np.isreal(r):
          #  print('ok2')
            if r>=z1min and r<=z1max:
             #   print('ok3')
                r = np.real(r)
                rho = get_rho_complete(N,z0,r,a)
              #  print(N,z0,r,a,rho)
                if rho>rho_max:
                #    print('ok4')
                    rho_max, z1 = rho, r
                elif rho==rho_max:
                 #   print('ok5')
                    z1 = min(z1,r)
    if z1<0 or rho_max<0:
        print(N,z0,a,roots,z1max)
        return None,None
    else:
        return z1,rho_max


def get_rho(N,W,nz0,nz1,z0,z1,xstar,weighted=False,verbose=False):
    z = nz0+nz1
    free = N-z
    # gurobi part
    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            p = m.addMVar((free,free), lb=0, ub=1)
            m.update()
            d = W.sum(axis=1)
            for i in range(free):
                m.addConstr(p[i,i]==0)
                for j in range(i):
                    constr = p[i,j]*(d[i]+d[j])
                    constr -= z1[i]+z1[j]
                    constr -= (z0[i]-z1[i])*xstar[j]
                    constr -= (z0[j]-z1[j])*xstar[i]
                    for k in range(N-z):
                        if k!=i and k!=j:
                            constr -= W[i,k]*p[j,k] + W[j,k]*p[i,k]
                    m.addConstr(constr==0)
                    m.addConstr(p[i,j]==p[j,i])
            m.setObjective(0)
            m.update()
            m.optimize()
            rho_guro = p.X
            rho = np.zeros((N,N))
            for k in range(nz0):
                rho[:free,free+k] = xstar
            for k in range(nz0,z):
                rho[:free,free+k] = 1-xstar
            rho[:free,:free] = rho_guro
            if weighted:
                rho = rho*W
                return rho.sum()/W.sum()
            else:
                A = (W>0).astype(int)
                rho = rho*A
                return rho.sum()/A.sum()
      
    
def get_rho_simu(N,nz0,nz1,W,max_time,weighted=False,burn=0,save_every=100,print_every=1000,timeit=False,just_mean=False):
    z = nz0+nz1
    free = N-z
    if weighted:
        A = W
    else:
        A = (W>0).astype(int)
    n_edges = A.sum()
    x = list(np.random.choice((0,1),size=N-z)) + [0]*nz0 + [1]*nz1
    rho_simu, times = list(), list()
    t = 0 # time tracking
    updated_yet = False
    c = 0 # a counter for save_every and print_every

    # iterate
    while t<max_time:
        if timeit:
            if c%print_every==0:
                sys.stdout.flush()
                sys.stdout.write(f"elapsed time {t:.3f}, max time {max_time:.3f}\r")
                
        # update time 
        waiting = np.random.exponential(1/free)
        t += waiting
                
        # update opinions
        u = np.random.randint(free)
        x_old = int(x[u])
        p = list(W[u])
        x_new = np.random.choice(x,p=p/sum(p))
        x[u] = x_new

        # update rho
        if t>burn:
            if not updated_yet:
                updated_yet = True
                n_active = 0
                for i in range(N):
                    for j in range(i):
                        if x[i]!=x[j]:
                            n_active += A[i,j]+A[j,i]
            else:
                if x_old!=x_new:
                    for j in range(N):
                        if x_new!=x[j]:
                            n_active = n_active+A[u,j]+A[j,u]
                        else:
                            n_active = n_active-A[u,j]-A[j,u]
            if c%save_every==0:
                rho_simu.append(n_active/n_edges)
                times.append(t)
        c += 1 # update coutner
         
    # end
    if just_mean:
        return np.mean(rho_simu)
    else:
        return rho_simu,times