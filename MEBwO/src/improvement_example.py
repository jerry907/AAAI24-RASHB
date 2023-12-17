import data.generation
import plot_settings
from meb.ball import MEBwO

import numpy as np
import matplotlib.pyplot as plt

n = 1000
d = 100

np.random.seed(3412)

datas = {
    "normal": data.generation.normal(0,1,n,d),
    "uniform_ball": data.generation.uniform_ball(n, d, 1),
    "uniform_ball_with_outliers": data.generation.uniform_ball_with_ouliters(n,d, 0.9, 1, 2, 3),
    "hyperspherical_shell": data.generation.hyperspherical_shell(n, d, 1, 2)
}

num_iter = 100

for name, data in datas.items():
    ball = MEBwO().fit(data, method="shenmaier", eta=0.9)
    inliers = [x for x in data if np.linalg.norm(x-ball.center) <= ball.radius]

    ball1 = MEBwO(center=ball.center, radius=ball.radius)
    ball2 = MEBwO(center=ball.center, radius=ball.radius)

    r1 = [ball.radius]
    r2 = [ball.radius]

    for _ in range(num_iter):
        ball1.improve(inliers, method="dcmeb")
        r1.append(ball1.radius)

        ball2.improve(inliers, method="dcssh", gamma=ball2.radius**2)
        r2.append(ball2.radius)

    plt.plot(range(num_iter+1), r1, color="r", label="DCMEB")
    plt.plot(range(num_iter+1), r2, color="b", label="DCSSH")
    plt.xlabel("Iterations")
    plt.ylabel("Radius")
    plt.legend()
    plt.savefig(r"images/improvement_example/improvement_example_{}".format(name), bbox_inches="tight")
    plt.clf()