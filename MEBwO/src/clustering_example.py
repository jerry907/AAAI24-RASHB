import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import data.generation

from meb.ball import MEBwO
from meb.geometry import M_estimate
import plot_settings

def plot_labelled(filename):
    for i in range(len(inliers_list)):
        inliers = np.array(inliers_list[i])
        plt.scatter(inliers[:,0], inliers[:,1], color=colours[i], label=i)

    plt.legend(title="Labels", ncol=2)
    plt.savefig(r"images/clustering/{}".format(filename), bbox_inches="tight")

n = 100
k = 10
num_balls = int(n/k)

colours = sns.color_palette("hls", num_balls)

np.random.seed(4444)
data = data.generation.normal(0, 5, n, 2)
not_labelled = data.copy()

fig, ax = plt.subplots(figsize=(10,10))
plt.gca().set_aspect('equal')

inliers_list = []
balls = []
for i in range(num_balls):
    print("Ball:\t{}".format(i))
    eta = k/len(not_labelled)

    # fit ball
    ball = MEBwO().fit(not_labelled, "shenmaier", eta=eta)
    inliers = [x for x in not_labelled if ball.contains(x)]
    inliers_list.append(inliers)
    for _ in range(100):
        ball.improve(inliers, method="dcssh", gamma=ball.radius**2)
    
    # remove data inside ball from set
    not_labelled = [x for x in not_labelled if not ball.contains(x)]

    # plot ball
    ax.add_patch(
                plt.Circle(ball.center, ball.radius, color=colours[i], fill=False)
            )
    
    balls.append(ball)

# plot labelled data
plot_labelled("with_balls")

# new plot
fig, ax = plt.subplots(figsize=(10,10))
plt.gca().set_aspect('equal')

# plot invisible balls for consistent plot size
for ball in balls:
    ax.add_patch(
            plt.Circle(ball.center, ball.radius, color=colours[i], fill=False, alpha=0)
        )

plot_labelled("no_balls")