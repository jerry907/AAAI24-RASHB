import numpy as np

# Functions to generate data from a specific distribution with given parameters, dimension, and number of points

def normal(mean, variance, n, d) -> np.array:
    """
    Generates n data points from Normal(mean, variance) distribution with given dimension
    Input:
        mean (float): mean of normal distribution
        variance (float): variance of normal distribution
        n (int): number of data points to generate
        d (int): dimension of the data points

    Return:
        data (np.array): generated data 
    """
    data = np.array([np.random.normal(mean, variance, d) for _ in range(n)])
    return data

def uniform(lb, ub, n, d) -> np.array:
    """
    Generates n data points from Uniform(lb,ub) distribution with given dimension
    Input:
        lb (float): lower bound of uniform distribution
        ub (float): upper bound of uniform distribution
        n (int): number of data points to generate
        d (int): dimension of the data points

    Return:
        data (np.array): generated data 
    """
    data = np.array([np.random.uniform(lb, ub, d) for _ in range(n)])
    return data

def poisson(lam, n, d) -> np.array:
    """
    Generates n data points from Poisson(lam) distribution with given dimension
    Input:
        lam (float): rate parameter of Poisson distribution
        n (int): number of data points to generate
        d (int): dimension of the data points

    Return:
        data (np.array): generated data 
    """
    data = np.array([np.random.poisson(lam, d) for _ in range(n)])
    return data

def check_c(c, d):
    """
    Utility function for generating spherical data
    If c is None (default), set it to d-dimensional 0 vector

    Input:
        c (array like): center
        d (int): dimension
    
    Return:
        c (array like): center
    """
    if c is None:
        c = np.zeros(d)
    
    return c

def point_on_hypersphere(d):
    """
    Generates a unit point on the surface of a unit (d-1)-dimension hypersphere

    Input:
        d (int): dimension
    
    Return:
        unit_point (np.array): generated point
    """
    point = np.random.normal(loc=0, scale=1, size=d)
    unit_point = point/np.linalg.norm(point)
    return unit_point

def point_in_ball(d, r, c):
    """
    Generates a uniform point inside the volume of a d-ball with center c and radius r

    Input:
        d (int): dimension
        r (float): radius
        c (array like): center
    
    Return:
        point (np.array): generated point
    """
    direction = point_on_hypersphere(d)
    magnitude = r*np.random.uniform(low=0, high=1)**(1/d)
    point = direction*magnitude + c

    return point

def uniform_ball(n, d, r, c=None):
    """
    Generates n points in a uniform d dimensional ball with centre c and radius r

    Input:
        n (int): number of points
        d (int): dimension
        r (float): radius of ball
        c (array like): center of ball
    
    Return:
        data (np.array): generated data
    """
    c = check_c(c, d)

    data = [point_in_ball(d, r, c) for _ in range(n)]

    return np.array(data)

def hyperspherical_shell(n, d, r1, r2, c=None):
    """
    Generates n points in a d-dimensional hyperspherical shell with centre c and inner radius r1 and outer radius r2

    Input:
        n (int): number of points
        d (int): dimension
        r1 (float): inner radius
        r2 (float): outer radius
        c (array like): center of hyperspherical shell

    Return:
        data (np.array): generated data
    """
    c = check_c(c, d)

    data = []

    while len(data) < n:
        point = point_in_ball(d, r2, c)
        if np.linalg.norm(point - c) > r1:
            data.append(point)
    
    return np.array(data)

def uniform_ball_with_ouliters(n, d, eta, r, r1, r2, c=None) -> np.array:
    """
    Generates n many data points where eta*n many are uniformly inside a ball B1 with centre c and radius r1,
    and (1-eta)*n many are uniformly inside a ball B2 with centre c and radius r2 but not in B1

    Input:
        n (int): total number of data points to generate
        d (int): dimension of data points
        eta (float): percentage of data points to be inside B1
        c (np.array): centre of ball
        r (float): radius of  ball
        r1 (float): inner radius of hyperspherical shell
        r2 (float): outer radius of hyperspherical shell
    
    Return:
        data (np.array): generated data
    """
    c = check_c(c,d)

    n_inner = int(np.floor(eta*n))
    n_outer = int(np.ceil((1-eta)*n))

    data_inner = uniform_ball(n_inner, d, r, c)
    data_outer = hyperspherical_shell(n_outer, d, r1, r2, c)

    data = np.vstack((data_inner, data_outer))
    np.random.shuffle(data)
    return np.array(data)

def two_clusters(n, d, eta, m1, v1, m2, v2) -> np.array:
    """
    Generates two clusters of normally distributed data with means and variances m1,m2, and v1,v2 respectively

    Input:
        n (int): total number of points to generate
        d (int): dimension of data
        eta (float): proportion of points to be in cluster 1 (with (1-eta)% in cluster 2)
        m1 (float): mean of normal distribution for cluster 1
        v1 (float): variance of normal distribution for cluster 1
        m1 (float): mean of normal distribution for cluster 2
        v1 (float): variance of normal distribution for cluster 2
    
    Return:
        data (np.array): generated data
    """
    k = int(np.floor(n*eta)) # number of points in cluster 1

    cluster1 = normal(m1, v1, k, d)
    cluster2 = normal(m2, v2, n-k, d)

    data = np.concatenate((cluster1, cluster2))

    return data