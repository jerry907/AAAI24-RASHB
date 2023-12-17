"""
Constructs plots of improvement heuristic performance benchmarks from logs
"""

import pandas as pd
import matplotlib.pyplot as plt

import plot_settings

heuristics = ["dcmeb", "dcssh"]
data_types = ["hyperspherical_shell", "normal", "uniform_ball", "uniform_ball_with_outliers"]
params = ["n", "d"]

colours = {"dcmeb": "r", "dcssh": "b"}
markers = {"dcmeb": "o", "dcssh": "^"}
axes = {"n": [500+500*i for i in range(10)], "d": [10+10*i for i in range(15)]}
fixed_params = {"n": "d100", "d": "n1000"}
xlabels = {
    "n": "$n$",
    "d": "$d$"
}

for data_type in data_types:
    for param in params:
        # load dataframes
        filename = "avg_pct_func_{0}_{1}_eta0p9_{2}".format(param, fixed_params[param], data_type)
        dfs = {heuristic: pd.read_csv(r"benchmarks/{0}/{1}/{2}.csv".format(heuristic, data_type, filename)) for heuristic in heuristics}
        

        # get avg% lists
        pct_lists = {heuristic: df["avg%"] for heuristic, df in dfs.items()}

        plt.figure()
        plt.xlabel(xlabels[param])
        plt.ylabel("avg%")
        
        for heuristic, pct_list in pct_lists.items():
            marker = markers[heuristic]
            colour = colours[heuristic]
            
            plt.plot(axes[param], pct_list, marker=marker, linestyle=":", color=colour, label=heuristic.upper())

        plt.legend()
        plt.savefig(r"images/improvement_r_benchmarks/{}".format(filename), bbox_inches="tight")

