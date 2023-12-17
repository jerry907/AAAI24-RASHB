import numpy as np
from . import geometry, gurobi_solvers

"""
Improvement heuristics for MEB

All functions should start with 'alg__' to be captured in algorithms dictionary (see bottom of algorithms.py)
"""

def alg__dcmeb(data, c, a=None, time_limit=None, log_file=""):
    """
    Direction-constrained MEB improvement heuristic

    Input:
        data (np.array): data set
        c (np.array): center of initial ball
        a (np.array): furthest point from center in data (direction vector)
        time_limit (float): time limit for solver (if not set then no limit)
        log_file (str): file location for logging (if not set then do not log)
        
    """
    if a is None:
        a = geometry.find_furthest(c, data)
    
    x, r, _ = gurobi_solvers.dc_meb(data, c, a, time_limit, log_file)

    new_c = c + x*(a-c)

    return new_c, r

def alg__dcssh(data, c, gamma, a=None, s=2):
    """
    Moves c in the direction c->a by 1/s times the minimum distance from 
    each point in data in the direction a->c to the surface of the ball

    Input:
        data (np.array): data set
        c (np.array): center of initial ball
        gamma (float): squared radius of initial ball
        a (np.array): furthest point from center in data (direction vector)
        s (float): scaling parameter for moving c along direction c->a

    Return:
        new_c (np.array): improved center
        new_r (float): improved radius
    """
    if a is None:
        a = geometry.find_furthest(c, data)
    
    beta = c - a
    x = min([geometry.Q(c=c, beta=beta, point=point, gamma=gamma) for point in data])

    new_c = c - (c-a)*x/2
    new_r = max([np.linalg.norm(new_c - point) for point in data])

    return new_c, new_r

# dictionary of functions whose name starts with "alg__" (i.e. the ones in this file)
algorithms = {name: func for name, func in locals().copy().items() if name.startswith("alg__")}