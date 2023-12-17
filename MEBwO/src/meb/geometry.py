import numpy as np

def find_furthest(p,data, core_set=None, return_index=False, find_closest=False) -> np.array:
    """
    Finds the furthest or closest point in data from p (l2 norm) that isn't in the core set if specified

    Input:
        p (array like): initial point
        data (array like): list of points to find furthest point from p
        core_set (array like): list of points in the core set not to consider
        return_index (bool): if True return the index of the furthest point, if False return the furthest point
        find_closest (bool): if True return closest point instead of furthest

    Return:
        point (array like): point in data which is furthest or closest to p
        OR
        index (int): index of points in data which is furthest or closest to p
    """
    if core_set is None:
        core = []
    else:
        core = core_set

    # initial values set to return point p if data is empty or only contains p
    dist = 0 # highest/lowest distance found so far
    point = p # furthest/closest point found so far
    index = 0 # index of point in data
    n = len(data)
    for i in range(n):
        x = data[i]
        if x not in core: # if no core set specified, always evaluates to True
            x_dist = np.linalg.norm(p-x)
            if (find_closest and x_dist < dist) or (not find_closest and x_dist > dist):
                dist = x_dist
                point = x
                index = i
    
    if return_index:
        return index
    else:
        return point

def diameter_approx(p, data, return_diameter=False):
    """
    Given an initial point p in data, finds q which is furthest from p and qdash which
    is furthest from q and if desired calculates diameter
    
    Input:
        p (array like): initial point
        data (array like): list of points to approximate diameter for
        return_diameter (bool): if True, also return distance (l2 norm) between q and qdash

    Return:
        out (tuple): tuple containing q, qdash, and if desired diameter
    """
    q = find_furthest(p, data)
    qdash = find_furthest(q, data)

    out = (q, qdash)
    if return_diameter:
        diameter = np.linalg.norm(q-qdash)
        out = out + (diameter,)
    
    return out

def mean_vector(data) -> np.array:
    """
    Calculates the mean vector from a set of d-dimensional vectors

    Input:
        data (array like): set of vectors to calculate the mean for

    Return:
        mean (np.array): mean vector
    """
    d = len(data[0])
    
    # calculate mean for each column in data
    # e.g.
    #   a = [
    #       [1,2],
    #       [3,4],
    #       [5,6]    
    #   ]
    #   calculate mean of [1,3,5], [2,4,6], return results as elements of array
    mean = np.array([
        np.mean([x[i] for x in data]) for i in range(d)
    ])
    return mean

def M_estimate(data):
    """
    Calculates all pairwise distances between points in the data and returns the largest distance
    WARNING: O(n^2)

    Input:
        data (array like): data
    Return:
        distance (float): largest pointwise distance between all points in the data
    """
    n = len(data)

    distance = max([
        max([np.linalg.norm(data[i] - x) for x in data]) for i in range(n)
    ])

    return distance

def k_closest(data, x, k) -> np.array:
    """
    Finds the k closest points to x in data

    Input:
        data (array like): data
        x (np.array): point to find k closest points in data to
        k (int): number of points closest to x to find
    
    Return:
        k_data (np.array): k points in data that are closest to x
        key_dist (float): maximum distance from x to points in k_data
    """
    n = len(data)
    # distance from x to every point
    distances = [np.linalg.norm(x-point) for point in data]

    sorted_distances = sorted(distances)

    # the distance where points with distances lower than this are the k closest
    key_dist = sorted_distances[k-1]

    k_data = np.array([data[i] for i in range(n) if distances[i] <= key_dist])

    return k_data, key_dist

def k_closest_cluster(data, x, k) -> np.array:
    """
    Finds the k closest points to x in data

    Input:
        data (array like): data
        x (np.array): point to find k closest points in data to
        k (int): number of points closest to x to find
    
    Return:
        k_data (np.array): k points in data that are closest to x
        key_dist (float): maximum distance from x to points in k_data
    """
    # distance from x to every point
    dis = np.linalg.norm(data - x, axis=-1)
    idx = np.argsort(dis)
    k_data = data[idx[:k]]
    # print("in k_closest_cluster number of inner points: ", len(k_data))
    key_dist = (dis[idx[:k]] ** 2).sum()
    return k_data, key_dist, dis[idx[k-1]]

def Q(c, beta, point, gamma) -> float:
    """
    Solves quadratic equation Q and returns x>=0

    Input:
        c (np.array): center of ball
        beta (np.array): direction from furthest point to c
        point (np.array): chosen point
        gamma (float): squared radius of ball
    
    Return:
        x (float): multiplier for direction to get from point a to the surface of the ball
    """
    alpha = point - c
    
    beta_beta = np.dot(beta, beta)
    alpha_beta = np.dot(alpha, beta)

    temp = alpha_beta**2 - beta_beta*(alpha@alpha - gamma)

    discriminant = np.sqrt(temp)

    x = (-alpha_beta + discriminant)/beta_beta

    return x