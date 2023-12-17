import numpy as np

import benchmarking.trials, benchmarking.utils
import data.loading
from meb import mebwo_algorithms

num_trials = 5

# data and M value pairs
data_types = {
    "uniform_ball": 2,
    "hyperspherical_shell": 4,
    "uniform_ball_with_outliers": 6,
    "normal": None,
    "mnist": None
}

# func name and func pairs
func_types = {
    "relaxation_heuristic": mebwo_algorithms.alg__relaxation_heuristic,
    "shrink": mebwo_algorithms.alg__shrink,
    "shrink_avg": mebwo_algorithms.alg__shrink_avg,
    "shenmaier": mebwo_algorithms.alg__shenmaier
}

param_types = {
    "n": False,
    "d": False,
    "eta": False
}

confirmed_funcs = benchmarking.utils.confirm_benchmark(func_types)
confirmed_data = benchmarking.utils.confirm_benchmark(data_types)

for param in param_types:
        msg = input("Run benchmarks for {}? (y/n) ".format(param))
        if msg == "y":
            param_types[param] = True

confirmed_params = {param: param_types[param] for param in param_types if param_types[param] == True}
print(confirmed_funcs.keys())
print(confirmed_data.keys())
print(confirmed_params.keys())
msg = input("Running {} benchmarks. Continue? (y/n) ".format(len(confirmed_funcs)*len(confirmed_data)*len(confirmed_params)))

if msg == "y":
    for func_name in confirmed_funcs:
        for data_type in confirmed_data:
            if param_types["n"]:
                n = [1000 + 3000*i for i in range(4,8)]
                d = 30
                eta = 0.9

                if func_name == "relaxation_heuristic":
                    M = confirmed_data[data_type]
                else:
                    M = None

                path = r"benchmarks/{0}/{1}/func_n_d{2}_eta{3}_{4}".format(func_name, data_type, d, benchmarking.utils.format_eta(eta), data_type)

                times = benchmarking.trials.run_trials_alg(
                    func=confirmed_funcs[func_name],
                    n=n,
                    d=d,
                    eta=eta,
                    num_trials=num_trials,
                    data_type=data_type,
                    log_file=r"{}.log".format(path),
                    M=M
                )

                title = benchmarking.utils.get_title(func_name, n, d, eta)
                benchmarking.utils.plot_times(
                    x_axis=n,
                    times=times,
                    xlabel="n",
                    ylabel="Time (s)",
                    title=title,
                    plot=False,
                    filepath=r"{}.png".format(path)
                )
            
            if param_types["d"]:
                n = 10000
                d = [10 + 10*i for i in range(10)]
                eta = 0.9

                if func_name == "relaxation_heuristic":
                    M = confirmed_data[data_type]
                else:
                    M = None
                
                path = r"benchmarks/{0}/{1}/func_d_n{2}_eta{3}_{4}".format(func_name, data_type, n, benchmarking.utils.format_eta(eta), data_type)

                times = benchmarking.trials.run_trials_alg(
                    func=confirmed_funcs[func_name],
                    n=n,
                    d=d,
                    eta=eta,
                    num_trials=num_trials,
                    data_type=data_type,
                    log_file=r"{}.log".format(path),
                    M=M
                )

                title = benchmarking.utils.get_title(func_name, n, d, eta)
                benchmarking.utils.plot_times(
                    x_axis=d,
                    times=times,
                    xlabel="d",
                    ylabel="Time (s)",
                    title=title,
                    plot=False,
                    filepath=r"{}.png".format(path)
                )
            
            if param_types["eta"]:
                n = 10000
                d = 30
                eta = [0.5+0.1*i for i in range(5)]

                if func_name == "relaxation_heuristic":
                    M = confirmed_data[data_type]
                else:
                    M = None

                path = r"benchmarks/{0}/{1}/func_eta_n{2}_d{3}_{4}".format(func_name, data_type, n, d, data_type)

                times = benchmarking.trials.run_trials_alg(
                    func=confirmed_funcs[func_name],
                    n=n,
                    d=d,
                    eta=eta,
                    num_trials=num_trials,
                    data_type=data_type,
                    log_file=r"{}.log".format(path),
                    M=M
                )

                title = benchmarking.utils.get_title(func_name, n, d, eta)
                benchmarking.utils.plot_times(
                    x_axis=eta,
                    times=times,
                    xlabel="eta",
                    ylabel="Time (s)",
                    title=title,
                    plot=False,
                    filepath=r"{}.png".format(path)
                )
                
    benchmarking.utils.notify()