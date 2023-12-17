import matplotlib.pyplot as plt

import plot_settings
import benchmarking.utils

data_types = ["hyperspherical_shell", "normal", "uniform_ball", "uniform_ball_with_outliers"]
params = ["n", "d"]

colours_data = {
    "hyperspherical_shell": "r",
    "normal": "b",
    "uniform_ball": "g",
    "uniform_ball_with_outliers": "m"
}

markers_data = {
    "hyperspherical_shell": "o",
    "normal": "^",
    "uniform_ball": "D",
    "uniform_ball_with_outliers": "s"
}

labels_data = {
    "hyperspherical_shell": "Hyp. Shell",
    "normal": "Normal",
    "uniform_ball": "Uniform Ball",
    "uniform_ball_with_outliers": "UBwO"
}

axes = {
    "n": [500+500*i for i in range(10)],
    "d": [50+10*i for i in range(10)],
}

fixed_params = {
    "n": "d100",
    "d": "n1000",
}

xlabels = {
    "n": "$n$",
    "d": "$d$"
}


for param in params:
    times_dict = {data_type: None for data_type in data_types}
    for data_type in data_types:
        filename = "func_{0}_{1}_eta0p9_{2}".format(param, fixed_params[param], data_type)
        times_dict[data_type] = benchmarking.utils.get_times_from_log(r"benchmarks/dcmeb/{0}/{1}.log".format(data_type, filename))
    
    plt.figure()
    plt.xlabel(xlabels[param])
    plt.ylabel("Time (s)")

    for data_type, times in times_dict.items():
        x_axis = axes[param]

        plt.plot(x_axis, times, marker=markers_data[data_type], color=colours_data[data_type], linestyle=":", label=labels_data[data_type])
    
    plt.legend()
    plt.savefig(r"images/dcmeb_benchmarks/{}".format(filename))
    plt.close()