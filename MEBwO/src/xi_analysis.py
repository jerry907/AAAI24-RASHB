import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from meb.meb_algorithms import alg__socp_heuristic
import plot_settings

import data.generation
from meb.gurobi_solvers import mebwo
from meb.geometry import M_estimate

n = 1000
d = 10
eta = 0.9

np.random.seed(1111) # reproducability
datas = {
    "normal": data.generation.normal(0,1,n,d),
    "uniform ball": data.generation.uniform_ball(n,d, r=1),
    "uniform ball with outliers": data.generation.uniform_ball_with_ouliters(n, d, eta, r=1, r1=2, r2=3),
    "hyperspherical shell": data.generation.hyperspherical_shell(n,d, r1=1, r2=2)
}

for name, data in datas.items():
    print(name)

    # get upper bound for M
    _, r, _ = alg__socp_heuristic(data, eps=1e-4)
    M_UB = 2*r

    # initialize sequence of M's
    num_trials = 50
    M_list = [M_UB*(1+k/2) for k in range(num_trials)]

    variances = {M: None for M in M_list}

    i = 0
    for M in M_list:
        print(i)
        i += 1

        # solve model and record variances
        _, _, xi, _ = mebwo(data=data, eta=eta, M=M, relax=True)
        variances[M] = np.var(xi)

    # plot
    sns.lineplot(x=M_list, y=variances.values(), label=name)

plt.xlabel("$M$")
plt.ylabel("Variance")
plt.legend()
plt.savefig(r"images/xi_analysis.png")
plt.show()