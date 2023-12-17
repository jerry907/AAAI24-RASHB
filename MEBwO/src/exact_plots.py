"""
Constructs plots of exact method benchmarks from logs
"""

import matplotlib.pyplot as plt

import benchmarking.utils
import plot_settings

params = ["n", "d", "eta", "M"]

axes = {
    "n": [300 + 50*i for i in range(10)],
    "d": [2 + 2*i for i in range(10)],
    "eta": [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
}

fixed_params = {
    "n": "d10",
    "d": "n300",
    "eta": "n300_d10",
    "M": "n300_d10"
}

xlabels = {
    "n": "$n$",
    "d": "$d$",
    "eta": "$\eta$",
    "M": "$M$"
}

for param in params:
    # construct filename
    filename = "func_{0}_{1}".format(param, fixed_params[param])
    if param != "eta":
        filename += "_eta0p9"
    filename += "_normal"
    
    # get times from log
    log_file = r"benchmarks/exact/normal/{}.log".format(filename)
    times = benchmarking.utils.get_times_from_log(log_file)
    if param == "M":
        axes[param] = benchmarking.utils.get_M_from_log(log_file)

    plt.figure()
    plt.xlabel(xlabels[param])
    plt.ylabel("Time (s)")

    if param == "d":
        plt.xticks(range(2, 22, 2))

    plt.plot(axes[param], times, marker="o", linestyle=":", mec="r", mfc="r")
    plt.savefig(r"images/exact_benchmarks/{}".format(filename), bbox_inches="tight")