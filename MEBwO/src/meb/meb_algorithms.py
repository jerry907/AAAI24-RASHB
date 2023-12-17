import numpy as np
from numpy.linalg.linalg import norm
from . import geometry, gurobi_solvers, ball

"""
Algorithms used to compute MEB

All functions should start with 'alg__' to be captured in algorithms dictionary (see bottom of algorithms.py)
"""

def point_union(X,p) -> np.array:
    """
    Returns the set union of the set X and the set {p}, i.e. if p in X then returns X, otherwise returns X U {p}

    Input:
        X (array like): set of points
        p (np.array): candidate point
    
    Return
        out (np.array): union of X and {p}
    """
    if p not in X:
        out = np.vstack((X,p))
    else:
        out = X
    
    return out

def alg__socp_exact(data): # solves the exact optimisation problem for MEB
        c, r = gurobi_solvers.meb(data)
        return c, r, None

def alg__socp_heuristic(data, eps): # algorithm 1 https://dl.acm.org/doi/10.1145/996546.996548
    p = data[0]
    q, qdash = geometry.diameter_approx(p, data)
    X = np.array([q,qdash])
    delta = eps/163

    while True: # might want to set a max number of iterations
        c, r, _ = gurobi_solvers.meb(X) # compute MEB(X)
        r_dash = r*(1+delta) # get radius for (1+delta) approximation to MEB(X)
        temp_ball = ball.Ball(c,r_dash*(1+eps/2)) # set temp ball

        if temp_ball.check_subset(data): # check if all the data is contained in temp ball
            break
        else:
            p = geometry.find_furthest(c, data) # p = argmax_(x\in S) [||c'-x||]
        
        X = np.vstack((X, p))

    return c, r, X

def psi(u, data):
    n = len(data) # number of points

    x = sum([
        u[i] * np.dot(data[i],data[i]) for i in range(n)
    ])
    y = np.array(
        sum([u[i] * data[i] for i in range(n)])
    )

    z = np.dot(y,y)

    out = x - z
    return out

def e(n, k) -> np.array:
    """
    Returns the standard basis vector e_k in R^n i.e. 0 n-vector with k-th element 1

    Input:
        n (int): dimension of vector
        k (int): element of vector to be 1
    
    Return:
        e (np.array): canonical basis vector
    """
    e = np.zeros(n)
    e[k] = 1
    return e

def alg__heuristic_1(data, eps): # algorithm 3.1 https://www.researchgate.net/publication/220133011_Two_Algorithms_for_the_Minimum_Enclosing_Ball_Problem
    alpha = geometry.find_furthest(data[0], data, return_index=True)
    beta = geometry.find_furthest(data[alpha], data, return_index=True)

    n = len(data)
    u = np.zeros(n)
    u[alpha] = 1/2
    u[beta] = 1/2

    X = np.array([data[alpha], data[beta]])
    c = sum([u[i]*data[i] for i in range(n)])
    gamma = psi(u, data)
    kappa = geometry.find_furthest(c, data, return_index=True)
    delta = ((np.linalg.norm(data[kappa]-c)**2)/gamma) - 1

    tol = (1+eps)**2 - 1

    while delta > tol:
        lam = delta/(2*(1+delta))
        u = (1-lam)*u + lam*e(n, kappa)
        c = (1-lam)*c + lam*data[kappa]
        X = point_union(X,data[kappa]) # X := X U {a^k}
        gamma = psi(u,data)
        kappa = geometry.find_furthest(c, data, return_index=True)
        delta = ((np.linalg.norm(data[kappa]-c)**2)/gamma) - 1

    return c, np.sqrt((1+delta)*gamma), X

def alg__heuristic_2(data, eps): # algorithm 4.1 https://www.researchgate.net/publication/220133011_Two_Algorithms_for_the_Minimum_Enclosing_Ball_Problem
    alpha = geometry.find_furthest(data[0], data, return_index=True)
    beta = geometry.find_furthest(data[alpha], data, return_index=True)

    n = len(data)
    u = np.zeros(n)
    u[alpha] = 1/2
    u[beta] = 1/2

    X = np.array([data[alpha], data[beta]])
    c = sum([u[i]*data[i] for i in range(n)])
    gamma = psi(u, data)
    kappa = geometry.find_furthest(c, data, return_index=True)
    xi = geometry.find_furthest(c, data, return_index=True, find_closest=True)
    delta_plus = ((np.linalg.norm(data[kappa]-c)**2)/gamma) - 1
    delta_minus = 1 - ((np.linalg.norm(data[xi]-c)**2)/gamma)
    delta = max(delta_plus, delta_minus)

    tol = (1+eps)**2 - 1

    while delta > tol:
        if delta > delta_minus:
            lam = delta/(2*(1+delta))
            u = (1-lam)*u + lam*e(n, kappa)
            c = (1-lam)*c + lam*data[kappa]
            X = point_union(X, data[kappa])
        else:
            temp = u[xi]/(1-u[xi])
            lam = min(delta_minus/(2*(1-delta_minus)), temp)
            if lam == temp:
                X = np.array([x for x in X if (x!=data[xi]).all()])
            else:
                pass
            u = (1+lam)*u - lam*e(n, xi)
            c = (1+lam)*c - lam*data[xi]
        
        gamma = psi(u, data)
        kappa = geometry.find_furthest(c, data, return_index=True)
        xi = geometry.find_furthest(c, data, return_index=True)
        delta_plus = ((np.linalg.norm(data[kappa]-c)**2)/gamma) - 1
        delta_minus = 1 - ((np.linalg.norm(data[xi] - c)**2)/gamma)
        delta = max(delta_plus, delta_minus)

    return c, np.sqrt((1+delta)*gamma), X

# dictionary of functions whose name starts with "alg_" (i.e. the ones in this file)
algorithms = {name: func for name, func in locals().copy().items() if name.startswith("alg_")}
