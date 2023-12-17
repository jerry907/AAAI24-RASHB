import numpy as np

import gurobipy as gp
from gurobipy import GRB

def meb(data, outputflag=0):
    """
    Solves the MEB problem using Gurobi

    Input:
        data (array like): list of data points to compute the MEB for

    Return:
        c_soln (np.array): center of the MEB
        r_soln (float): radius of the MEB
    """
    n = len(data) # number of points
    d = len(data[0]) # dimension TODO: make this better

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', outputflag)
        env.start()
        with gp.Model(env=env) as m:
            c = m.addMVar(shape=d, lb=-GRB.INFINITY, name="center")
            r = m.addVar(name="radius")

            m.setObjective(r, GRB.MINIMIZE)

            #TODO: find a way to vectorise this
            for i in range(n):
                m.addMQConstr(
                    Q=np.identity(d),
                    c=np.append(-2*data[i],-1),
                    sense=GRB.LESS_EQUAL,
                    rhs=-1*(data[i]@data[i]),
                    xQ_L=c,
                    xQ_R=c,
                    xc=c.tolist().append(r)
                )
            
            m.optimize()

            c_soln = [v.x for v in c.tolist()]
            r_soln = np.sqrt(m.getVarByName(name="radius").x)

    return c_soln, r_soln, None

def mebwo(data, eta, M, relax=False, time_limit=None, log_file="", outputflag=0):
    """
    Solves the MEBwO problem for eta% of the points covered using Gurobi

    Input:
        data (array like): list of data points to compute the MEB for
        eta (float): percentage of points to be covered, i.e. eta=0.9 means 90% of points in data are inside the ball
        M (float): value of M for big M constraint
        relax (bool): if True then relax binary variables to 0 <= xi[i] <= 1 for all i
        time_limit (float): time limit for solver (if not set then no limit)
        log_file (str): file location for logging (if not set then do not log)

    Return:
        c_soln (np.array): center of the MEB
        r_soln (float): radius of the MEB
        xi_soln (list): binary variables for outlier or not outlier
        m.Runtime (float): time taken to solve model
    """
    n = len(data) # number of points
    d = len(data[0]) # dimension
    k = int(np.ceil(n*(1-eta))) # number of points that are outliers
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', outputflag)
        env.start()
        with gp.Model(env=env) as m:
            if log_file != "":
                m.setParam(GRB.Param.LogFile, log_file)

            if time_limit is not None:
                m.setParam(GRB.Param.TimeLimit, time_limit)

            c = m.addMVar(shape=d, lb=-GRB.INFINITY, name="center")
            gamma = m.addVar(name="radius")

            if relax:
                xi = m.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
            else:
                xi = m.addVars(n, vtype=GRB.BINARY)

            m.setObjective(gamma, GRB.MINIMIZE)

            for i in range(n):
                m.addMQConstr(
                    Q=np.identity(d),
                    c=-1*np.concatenate((2*data[i], [M,1])),
                    sense=GRB.LESS_EQUAL,
                    rhs=-1*(data[i]@data[i]),
                    xQ_L=c,
                    xQ_R=c,
                    xc=c.tolist() + [xi[i], gamma]
                )
            
            m.addConstr(gp.quicksum(xi[i] for i in range(n)) <= k)

            m.optimize()

            c_soln = [v.x for v in c.tolist()]
            r_soln = np.sqrt(gamma.x)
            xi_soln = [xi[i].x for i in range(n)]
            runtime = m.Runtime

    return c_soln, r_soln, xi_soln, runtime

def dc_meb(data, c, a, time_limit=None, log_file="", outputflag=0):
    """
    Solves the direction-constrained MEB problem

    Input:
        data (np.array): data set
        c (np.array): center of initial ball
        a (np.array): furthest point from center in data (direction vector)
        time_limit (float): time limit for solver (if not set then no limit)
        log_file (str): file location for logging (if not set then do not log)
    
    Return:
        x_soln (float): proportion along direction c->a to move c
        r_soln (float): new radius
    """
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', outputflag)
        env.start()
        with gp.Model(env=env) as m:
            if log_file != "":
                m.setParam(GRB.Param.LogFile, log_file)

            if time_limit is not None:
                m.setParam(GRB.Param.TimeLimit, time_limit)

            x = m.addVar()
            gamma = m.addVar()

            m.setObjective(gamma, GRB.MINIMIZE)

            alphas = [c-point for point in data]
            beta = a-c

            beta_beta = beta@beta

            for alpha in alphas:
                m.addQConstr(
                    lhs=beta_beta*(x*x) + 2*x*(alpha@beta) + (alpha@alpha),
                    sense=GRB.LESS_EQUAL,
                    rhs=gamma
                )

            m.optimize()

            x_soln = x.x
            r_soln = np.sqrt(gamma.x)
            runtime = m.Runtime

    return x_soln, r_soln, runtime