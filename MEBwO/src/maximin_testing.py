import numpy as np
import matplotlib.pyplot as plt

from data import generation
from meb import meb_algorithms
from meb import geometry

def alg__maximin(data, eta, eps):
    """
    
    """
    num_outliers = int(np.ceil((1-eta)*len(data)))

    c_dash, _, _ = meb_algorithms.alg__socp_heuristic(data, eps)

    outliers = [geometry.find_furthest(c_dash, data)]
    inliers = [x for x in data if not np.array_equal(x, outliers[0])]

    for _ in range(num_outliers-1):
        distances = []
        for x in inliers:
            distances.append(
                min([np.linalg.norm(x-y) for y in outliers])
            )
        
        max_dist_index = np.argmax(distances)
        new_outlier = inliers[max_dist_index]
        outliers.append(new_outlier)
        inliers = [x for x in inliers if not np.array_equal(x, new_outlier)]
    
    c, r, _ = meb_algorithms.alg__socp_heuristic(inliers, eps)

    return c, r, inliers, outliers

data = generation.uniform_ball_with_ouliters(n=100,d=2,eta=0.9,c=[0,0],r1=1,r2=3,sep=1)

c, r, inliers, outliers = alg__maximin(data, 0.9, 1e-2)

n = len(outliers)
x_outliers = [outliers[i][0] for i in range(n)]
y_outliers = [outliers[i][1] for i in range(n)]

m = len(inliers)
x_inliers = [inliers[i][0] for i in range(m)]
y_inliers = [inliers[i][1] for i in range(m)]

fig,ax = plt.subplots()
plt.gca().set_aspect('equal')

plt.scatter(x_outliers, y_outliers, color="g", label="outliers")
plt.scatter(x_inliers, y_inliers, color="b", label="inliers")
plt.scatter(c[0], c[1], color="red", marker="x", label="center")
ax.add_patch(
    plt.Circle(c, r, color="red", fill=False, label="ball")
)
plt.legend()
plt.show()