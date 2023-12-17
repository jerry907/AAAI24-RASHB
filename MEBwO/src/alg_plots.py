"""
Constructs plots of construction algorithms benchmarks from logs
"""

import matplotlib.pyplot as plt

import plot_settings
import benchmarking.utils

heuristics = ["relaxation_heuristic", "shenmaier", "shrink", "shrink_avg"]
data_types = ["hyperspherical_shell", "normal", "uniform_ball", "uniform_ball_with_outliers"]
params = ["n", "d", "eta"]

colours_funcs = {
    "relaxation_heuristic": "r",
    "shenmaier": "b",
    "shrink": "g",
    "shrink_avg": "m"
}

colours_data = {
    "hyperspherical_shell": "r",
    "normal": "b",
    "uniform_ball": "g",
    "uniform_ball_with_outliers": "m"
}

markers_funcs = {
    "relaxation_heuristic": "o",
    "shenmaier": "^",
    "shrink": "D",
    "shrink_avg": "s"
}

markers_data = {
    "hyperspherical_shell": "o",
    "normal": "^",
    "uniform_ball": "D",
    "uniform_ball_with_outliers": "s"
}

labels_funcs = {
    "relaxation_heuristic": "RBH",
    "shenmaier": "Shenmaier",
    "shrink": "Shrink",
    "shrink_avg": "APH"
}

labels_data = {
    "hyperspherical_shell": "Hyp. Shell",
    "normal": "Normal",
    "uniform_ball": "Uniform Ball",
    "uniform_ball_with_outliers": "UBwO"
}

axes = {
    "n": [1000 + 3000*i for i in range(10)],
    "d": [10 + 10*i for i in range(10)],
    "eta": [0.5+0.1*i for i in range(5)]
}

fixed_params = {
    "n": "d30",
    "d": "n10000",
    "eta": "n10000_d30"
}

xlabels = {
    "n": "$n$",
    "d": "$d$",
    "eta": "$\eta$"
}

for data_type in data_types:
    for param in params:
        if data_type == "normal" and param == "n":
            # construct filename
            filename = "func_{0}_{1}".format(param, fixed_params[param])
            if param != "eta":
                filename += "_eta0p9"
            filename += "_{}".format(data_type)

            # get times from log
            times_dict = {heuristic: benchmarking.utils.get_times_from_log(r"benchmarks/{0}/{1}/{2}.log".format(heuristic, data_type, filename)) for heuristic in heuristics}
            results_dict = {heuristic: benchmarking.utils.get_r_from_log(r"benchmarks/{0}/{1}/{2}.log".format(heuristic, data_type, filename)) for heuristic in heuristics}

            # === times
            
            plt.figure()
            plt.xlabel(xlabels[param])
            plt.ylabel("Time (s)")

            for heuristic, times in times_dict.items():
                if heuristic != "shrink_avg":
                    if heuristic == "shenmaier" and param == "n":
                        x_axis = [1000 + 3000*i for i in range(8)]
                    elif heuristic == "relaxation_heuristic" and data_type == "uniform_ball_with_outliers" and param == "d":
                        x_axis = [10 + 10*i for i in range(8)]
                    else:
                        x_axis = axes[param]

                    plt.plot(x_axis, times, marker=markers_funcs[heuristic], color=colours_funcs[heuristic], linestyle=":", label=labels_funcs[heuristic])
                
            if data_type == "normal" and param == "eta":
                plt.legend(loc="best", bbox_to_anchor=(0.5,0.,0.5,0.5))
            else:
                plt.legend()

            plt.savefig("images/alg_benchmarks/by_data/{}".format(filename), bbox_inches="tight")
            plt.close()
            
            # === results
            plt.figure()
            plt.xlabel(xlabels[param])
            plt.ylabel("Radius")

            for heuristic, results in results_dict.items():
                if heuristic == "shenmaier" and param == "n":
                    x_axis = [1000 + 3000*i for i in range(8)]
                elif heuristic == "relaxation_heuristic" and data_type == "uniform_ball_with_outliers" and param == "d":
                    x_axis = [10 + 10*i for i in range(8)]
                else:
                    x_axis = axes[param]
                
                plt.plot(x_axis, results, marker=markers_funcs[heuristic], color=colours_funcs[heuristic], linestyle=":", label=labels_funcs[heuristic])
            
            if data_type == "normal" and param == "n":
                plt.legend(loc=2, bbox_to_anchor=(0,0.75))
            else:
                plt.legend()
            plt.savefig("images/alg_benchmarks/by_data/{}_r".format(filename), bbox_inches="tight")
            plt.close()


for heuristic in heuristics:
    for param in params:
        # construct filename
        filename = "func_{0}_{1}".format(param, fixed_params[param])
        if param != "eta":
            filename += "_eta0p9"

        # get times from log
        times_dict = {data_type: benchmarking.utils.get_times_from_log(r"benchmarks/{0}/{1}/{2}_{3}.log".format(heuristic, data_type, filename, data_type)) for data_type in data_types}
        
        plt.figure()
        plt.xlabel(xlabels[param])
        plt.ylabel("Time (s)")

        for data_type, times in times_dict.items():
            if heuristic == "shenmaier" and param == "n":
                x_axis = [1000 + 3000*i for i in range(8)]
            elif heuristic == "relaxation_heuristic" and data_type == "uniform_ball_with_outliers" and param == "d":
                x_axis = [10 + 10*i for i in range(8)]
            else:
                x_axis = axes[param]

            
            plt.plot(x_axis, times, marker=markers_data[data_type], color=colours_data[data_type], linestyle=":", label=labels_data[data_type])
        
        if heuristic == "relaxation_heuristic" and param == "eta":
            plt.legend(loc="best", bbox_to_anchor=(0.45,0.45))
        else:
            plt.legend()
        
        plt.savefig(r"images/alg_benchmarks/by_func/{0}_{1}".format(heuristic, filename), bbox_inches="tight")
        plt.close()
