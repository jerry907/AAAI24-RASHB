import numpy as np
import timeit
import pandas as pd
from meb import mebwo_algorithms
from meb import geometry

from meb.geometry import find_furthest, k_closest
from meb.gurobi_solvers import dc_meb
from meb.improvement_algorithms import alg__dcmeb, alg__dcssh

from meb.meb_algorithms import alg__socp_heuristic
from meb.mebwo_algorithms import alg__relaxation_heuristic, alg__shenmaier, alg__shrink_avg

from . import utils
import meb
import data

import benchmarking

# this file is a mess dont look at it

def run_trials_exact(n, d, eta, num_trials, data_, log_file=None, data_file=None):
    """
    Runs trials for the exact solver for MEBwO and returns averaged runtimes
    One of n, d, eta should be a list for benchmarking on that parameter

    Input:
        n (int): number of data points (rows)
        d (int): dimension of data (columns)
        eta (float): proportion of data covered by MEBwO
        num_trials (int): number of trials to run for each experiment (for averaging)
        data_ (np.array): data set
        log_file (str) (default None): file path for log file (if None, no logging)
        data_file (str): file path for original data (for logging)
    
    Return:
        avg_times (list of floats): average runtime for each experiment
    """
    # store variables by references in params dictionary
    params = {"n": n, "d": d, "eta": eta}
    
    trial_param, trial_param_vals = utils.find_trial_param

    utils.check_log(log_file)

    data_shape = data_.shape
    num_rows = data_shape[0]
    num_columns = data_shape[1]

    times = []

    for x in trial_param_vals:
        # update trial parameter in vars dictionary
        params[trial_param] = x

        # update all parameters (will be unchanged except for the trial parameter)
        n_ = params["n"]
        d_ = params["d"]
        eta_ = params["eta"]

        # elapsed time for each trial
        trials = [0]*num_trials

        for i in range(num_trials):
            utils.progress_report(x, trial_param, i)

            # get combination of rows/columns
            rows = np.random.choice(num_rows, size=n_, replace=False)
            columns = np.random.choice(num_columns, size=d_, replace=False)

            # load subset of data and calculate M
            trial_data = data.loading.subset_data(data_, rows, columns)
            M = meb.geometry.M_estimate(trial_data)

            # solve model and store variables
            c, r, _, trials[i] = meb.gurobi_solvers.mebwo(trial_data, eta_, M, log_file=log_file)
            
            if log_file is not None:
                utils.benchmark_logger(filepath=log_file, elapsed=trials[i], n=n_, d=d_, eta=eta_, M=M, r=r, c=c, trial_number=i, num_trials=num_trials, data_filepath=data_file, rows=rows, columns=columns)
        
        times.append(trials)
    
    avg_times = utils.calc_avg_times(times)

    return avg_times

def run_trials_alg(func, n, d, eta, num_trials, data_type, log_file=None, **kwargs):
    params = {"n": n, "d": d, "eta": eta}

    trial_param, trial_param_vals = utils.find_trial_param(params)

    utils.check_log(log_file)

    times = []

    for x in trial_param_vals:
        # update trial parameter in vars dictionary
        params[trial_param] = x

        # update all parameters (will be unchanged except for the trial parameter)
        n_ = params["n"]
        d_ = params["d"]
        eta_ = params["eta"]

        # elapsed time for each trial
        trials = [0]*num_trials

        for i in range(num_trials):
            utils.progress_report(x, trial_param, i)

            # load data
            filename = r"datasets/{0}/{1}_n{2}_d{3}".format(data_type, data_type, n_, d_)
            if data_type == "uniform_ball_with_outliers":
                filename += r"_eta{}".format(utils.format_eta(eta_))
            filename += r"_{}.csv".format(i)
            data_ = data.loading.from_csv(filename)

            # only need to calculate M when data is normal
            if data_type == "normal" and func == meb.mebwo_algorithms.alg__relaxation_heuristic:
                _, r, _ = alg__socp_heuristic(data_, eps=1e-4)
                kwargs["M"] = 2*r

            start = timeit.default_timer()
            c, r, _ = func(data_, eta_, **kwargs)
            elapsed = timeit.default_timer() - start

            trials[i] = elapsed

            if log_file is not None:
                if func in [meb.mebwo_algorithms.alg__shrink, meb.mebwo_algorithms.alg__shenmaier]:
                    kwargs["M"] = None

                utils.benchmark_logger(filepath=log_file, elapsed=trials[i], n=n_, d=d_, eta=eta_, M=kwargs["M"], r=r, c=c, trial_number=i, num_trials=num_trials, data_filepath=filename)

        times.append(trials)
    
    avg_times = utils.calc_avg_times(times)

    return avg_times

def run_trials_improvement_r(n, d, num_trials, num_iter, data_type, log_file=None, **kwargs):
    params = {"n": n, "d": d}
    eta = 0.9

    trial_param, trial_param_vals = utils.find_trial_param(params)

    utils.check_log(log_file)

    results_dcmeb = []
    results_dcssh = []

    for x in trial_param_vals:
        params[trial_param] = x

        n_ = params["n"]
        d_ = params["d"]
    
        result_dcmeb = [x]
        result_dcssh = [x]

        for i in range(num_trials):
            utils.progress_report(x, trial_param, i)

            # load data
            filename = r"datasets/improvement_datasets/{0}/{1}_n{2}_d{3}".format(data_type, data_type, n_, d_)
            if data_type == "uniform_ball_with_outliers":
                filename += r"_eta{}".format(utils.format_eta(eta))
            filename += r"_{}.csv".format(i)
            data_ = data.loading.from_csv(filename)

            # solve shenmaier
            c, r, _ = mebwo_algorithms.alg__shenmaier(data_, eta)
            result_dcmeb.append(r)
            result_dcssh.append(r)

            # get inliers
            data_ = [x for x in data_ if np.linalg.norm(c-x) <= r]

            # improvement heuristic
            for _ in range(num_iter):
                c_dcmeb, r_dcmeb = alg__dcmeb(data_, c)
                
                c_dcssh, r_dcssh = alg__dcssh(data_, c, r**2)

            result_dcmeb.append(r_dcmeb)
            result_dcssh.append(r_dcssh)
        
        results_dcmeb.append(result_dcmeb)
        results_dcssh.append(result_dcssh)
    
    # format dataframe
    columns = [trial_param]
    for i in range(num_trials):
        columns.append("r{}".format(i+1))
        columns.append("r{}^".format(i+1))
    
    results_dcmeb_df = pd.DataFrame(results_dcmeb, columns=columns)
    results_dcssh_df = pd.DataFrame(results_dcssh, columns=columns)

    for results_df in [results_dcmeb_df, results_dcssh_df]:
        pcts = []
        for index, row in results_df.iterrows():
            pct_list = [0]*num_trials
            for i in range(num_trials):
                pct_list[i] = (1 - (row["r{}^".format(i+1)]/row["r{}".format(i+1)]))*100
            pct = np.mean(pct_list)
            pcts.append(pct)
    
        results_df["avg%"] = pcts
        
    return results_dcmeb_df, results_dcssh_df

def run_trials_improvement_time(n, d, num_trials, data_type, log_file=None, **kwargs):
    params = {"n": n, "d": d}
    eta = 0.9

    trial_param, trial_param_vals = utils.find_trial_param(params)

    utils.check_log(log_file)

    times = []

    for x in trial_param_vals:
        params[trial_param] = x

        n_ = params["n"]
        d_ = params["d"]
    
        trials = [0]*num_trials

        for i in range(num_trials):
            utils.progress_report(x, trial_param, i)

            # load data
            filename = r"datasets/improvement_datasets/{0}/{1}_n{2}_d{3}".format(data_type, data_type, n_, d_)
            if data_type == "uniform_ball_with_outliers":
                filename += r"_eta{}".format(utils.format_eta(eta))
            filename += r"_{}.csv".format(i)
            data_ = data.loading.from_csv(filename)

            # solve shenmaier
            c, r, _ = mebwo_algorithms.alg__shenmaier(data_, eta)

            # get inliers
            inliers = [x for x in data_ if np.linalg.norm(c-x) <= r]
            a = geometry.find_furthest(c, inliers)

            c, r, trials[i] = dc_meb(data_, c, a, log_file=log_file, outputflag=1)

            if log_file is not None:
                utils.benchmark_logger(filepath=log_file, elapsed=trials[i], n=n_, d=d_, eta=None, M=None, r=r, c=c, trial_number=i, num_trials=num_trials, data_filepath=filename)

        times.append(trials)

    avg_times = utils.calc_avg_times(times)
        
    return avg_times

def mnist_benchmark(df, func, number, eta, log_file=None):
    """
    Calculates F1 score and records runtime

    Input:
        df (pd.DataFrame): MNIST data
        func (function): function for fitting MEBwO (shenmaier or shrink_avg)
        number (int): which number to fit MEBwO to
        eta (float): percentage of inliers
        log_file (str): log file
    """
    # construct dataframes
    df_inliers = df[df[0] == number]
    df_outliers = df[df[0] != number]

    # sample outliers
    n = len(df_inliers)
    k = int(np.floor((1-eta)*n))
    outliers = df_outliers.sample(n=k)

    # add outliers to inliers
    df_all = pd.concat([df_inliers, outliers])
    df_all = df_all.sample(frac=1) # shuffle
    labels = df_all[0]

    # convert data to array for fitting MEBwO
    data = df_all.drop(labels=0, axis=1).to_numpy() # array

    bar = "===================="
    print(bar)
    print("PROGRESS")
    print("\tNumber:\t{}".format(number))
    print("\teta:\t{}".format(eta))
    print(bar)
    # fit and time MEBwO
    if func == alg__relaxation_heuristic:
        _, r_ball, _ = alg__socp_heuristic(data, eps=1e-4)
        M = 2*r_ball
        outputflag = 1
        print(M)
    else:
        M = None

    start = timeit.default_timer()
    c, r, _ = func(data, eta, M, outputflag)
    elapsed = timeit.default_timer() - start

    # calculate F1 score
    TP = 0
    FP = 0
    FN = 0
    for label, x in zip(labels, data):
        if np.linalg.norm(c-x) <= r: # inside ball
            if label == number:
                TP += 1
            else:
                FP += 1
        else: # outside ball
            if label == number:
                FN += 1
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(precision*recall)/(precision+recall)

    if log_file is not None:
        utils.mnist_logger(filepath=log_file, num=number, eta=eta, F1=F1, elapsed=elapsed)

    return F1, elapsed
