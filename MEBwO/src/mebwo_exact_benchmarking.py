import numpy as np

import benchmarking.trials, benchmarking.utils
import data.loading
from meb import gurobi_solvers, geometry

num_trials = 5

data_types = ["normal"]

np.random.seed(1234)

for data_type in data_types:
    data_filepath = r"datasets/{}.csv".format(data_type)
    data_set = data.loading.from_csv(data_filepath)

    if False:
        n = [300 + 50*i for i in range(10)]
        eta = 0.9
        d = 10

        file_name = r"{0}/func_n_d{1}_eta{2}_{3}".format(data_type, d, benchmarking.utils.format_eta(eta), data_type)
        log_file = r"benchmarks/exact/{0}.log".format(file_name)

        times = benchmarking.trials.run_trials_exact(n, d, eta, num_trials, data_set, log_file=log_file, data_file=data_filepath)

        benchmarking.utils.plot_times(
            x_axis=n,
            times=times,
            xlabel="n",
            ylabel="Time",
            title="Running time for MEBwO as a function of n, d={0}, eta={1}".format(d, eta),
            plot=False,
            filepath=r"benchmarks/exact/{0}.png".format(file_name)
        )


    if False:
        n = 300
        eta = 0.9
        d = [2 + 2*i for i in range(10)]

        file_name = r"{0}/func_d_n{1}_eta{2}_{3}".format(data_type, n, benchmarking.utils.format_eta(eta), data_type)
        log_file = r"benchmarks/exact/{0}.log".format(file_name)

        times = benchmarking.trials.run_trials_exact(n, d, eta, num_trials, normal_data, log_file=log_file, data_file=normal_filepath)
        
        benchmarking.utils.notify()
        
        benchmarking.utils.plot_times(
            x_axis=d,
            times=times,
            xlabel="d",
            ylabel="Time",
            title="Running time for MEBwO as a function of d, n={}, eta={}".format(n, eta),
            plot=True,
            filepath=r"benchmarks/exact/{0}.png".format(file_name)
        )

    if False:
        n = 300
        d = 10
        eta = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

        file_name = r"{0}/func_eta_n{1}_d{2}_{3}".format(data_type, n, d, data_type)
        log_file = r"benchmarks/exact/{0}.log".format(file_name)

        times = benchmarking.trials.run_trials_exact(n, d, eta, num_trials, data_set, log_file=log_file, data_file=data_filepath)

        benchmarking.utils.plot_times(
            x_axis=eta,
            times=times,
            xlabel="eta",
            ylabel="Time",
            title="Running time for MEBwO as a function of eta, n={}, d={}".format(n, d),
            plot=False,
            filepath=r"benchmarks/exact/{0}.png".format(file_name)
        )

    if False:
        times = []

        n = 300
        d = 10
        eta = 0.9

        file_name = r"{0}/func_M_n{1}_d{2}_eta{3}_{4}".format(data_type, n, d, benchmarking.utils.format_eta(eta), data_type)
        log_file = r"benchmarks/exact/{0}.log".format(file_name)

        rows = range(n)
        columns = range(d)

        exp_data = data.loading.subset_data(normal_data, rows, columns)

        M_UB = geometry.M_estimate(exp_data)
        M_list = [M_UB*i for i in range(1,11)]

        file_name = r"{0}/func_M_n{1}_d{2}_eta{3}_{4}".format(data_type, n, d, benchmarking.utils.format_eta(eta), data_type)
        log_file = r"benchmarks/exact/{0}.log".format(file_name)

        benchmarking.utils.check_log(log_file)

        data_shape = normal_data.shape
        num_rows = data_shape[0]
        num_columns = data_shape[1]

        for M in M_list:
            trials = [0]*num_trials
            for i in range(num_trials):
                benchmarking.utils.progress_report(M, "M", i)

                c, r, xi, trials[i] = gurobi_solvers.mebwo_exact(exp_data, eta, M, log_file=log_file)

                benchmarking.utils.benchmark_logger(filepath=log_file, elapsed=trials[i], n=n, d=d, eta=eta, M=M, r=r, c=c, xi=xi, trial_number=i, num_trials=num_trials, data_filepath=normal_filepath, rows=rows, columns=columns)

            times.append(trials)
        
        avg_times = benchmarking.utils.calc_avg_times(times)

        benchmarking.utils.plot_times(
            x_axis=M_list,
            times=avg_times,
            xlabel="M",
            ylabel="Time",
            title="Running time for MEBwO as a function of M, n={}, d={}, eta={}".format(n, d, eta),
            plot=True,
            filepath=r"images\benchmarks\mebwo_runtimes_M.png"
        )

benchmarking.utils.notify()