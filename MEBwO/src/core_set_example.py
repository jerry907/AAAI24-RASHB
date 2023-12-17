import numpy as np
import matplotlib.pyplot as plt

from meb.ball import MEB
import data.generation
import plot_settings

n = 1000
d = 2

np.random.seed(1000)
data = data.generation.normal(0, 1, n, d)
ball = MEB().fit(data, method="socp_heuristic", eps=1e-4)

ball.plot(data, show=False, figsize=8.5)
plt.legend()
plt.savefig(r"images/core_set_example.png", bbox_inches='tight')
plt.show()