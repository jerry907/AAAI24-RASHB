import numpy as np
import xpress as xp

xp.controls.outputlog = 0

def socp_solver(data):
    """
    Solves the MEB problem using Xpress

    Input:
        data (array like): list of data points to compute the MEB for

    Return:
        c_soln (NumPy array): center of the MEB
        r_soln (float): radius of the MEB
    """
    n = len(data) # number of points
    d = len(data[0]) # dimension TODO: make this better

    m = xp.problem(name="MEB") # problem object

    r = xp.var(name="radius") # radius dv
    c = np.array([xp.var(name="c_{}".format(i), lb=-xp.infinity) for i in range(d)]) # center dv's

    m.addVariable(r, c)
    m.setObjective(r)
    
    m.addConstraint(xp.sqrt(xp.Dot(c-data[i], c-data[i])) - r <= 0 for i in range(n)) # norm constraints

    m.solve()
    c_soln = m.getSolution(c)
    r_soln = m.getSolution(r)

    return c_soln, r_soln