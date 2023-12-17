import time
import logging
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def calc_avg_times(avg_times) -> list:
    """
    Takes a list of list in the form [ [*], [*], ...] and returns a list with the average of each sublist

    Input:
        avg_times (list of list of floats): list of lists of times to find average times for

    Return:
        times (list of floats): list of average times
    """
    times = []
    for lst in avg_times:
        times.append(np.mean(lst))
    
    return times

def plot_times(x_axis, times, xlabel, ylabel, title, plot, filepath=None) -> None:
    """
    Creates a plot for benchmarking

    Input:
        x_axis (array like): x values for points to plot
        times (array like): y values (runtimes) to plot
        xlabel (str): x axis label
        ylabel (str): y axis label
        title (str): title of plot
        plot (bool): if True, display the plot
        filepath (str/None) (optonal): if not None will save the plot to the specified filepath
    
    Return:
        None
    """
    plt.figure()
    plt.plot(x_axis, times, marker="o", linestyle=":", mec="r", mfc="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if filepath is not None:
        plt.savefig(filepath)
    
    if plot:
        plt.show()
    
    return None

def progress_report(x, param_name, i) -> None:
    """
    Prints the current step in the benchmarking process

    Input:
        x (float): the parameter being benchmarked (i.e. number of points/dimension)
        param_name (str): name of the parameter being benchmarked
        i (int): current trial of x
    
    Return:
        None
    """
    bar = "===================="
    print(bar)
    print("PROGRESS:")
    print("\t{0}:\t{1}".format(param_name, x))
    print("\tTrial:\t{}".format(i+1))
    print(bar)

    return None

def reset_log(filepath):
    fileh = logging.FileHandler(filepath, "a")
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fileh.setFormatter(formatter)
    
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    log.addHandler(fileh)
    log.setLevel("INFO")
    return None

def benchmark_logger(filepath, elapsed, n, d, eta, M, r, c, trial_number, num_trials, data_filepath, rows=None, columns=None):
    """
    After a trial has been completed, logs details to the specified file

    Input:
        filepath (str): filepath of the log file to be written to
        elapsed (float): elapsed runtime of trial
        n (int): number of rows used in data
        d (int): dimension of data
        eta (float): proportion of points covered by the MEBwO
        M (float): big M parameter for solving exact model
        r (float): solution for radius
        c (list): solution for center
        trial_number (int): current trial in experiment
        num_trials (int): total number of trials in experiment
        data_filepath (str): filepath of data used
        rows (list): rows in data that have been used
        columns (list): columns in data that have been used
    
    Return:
        None
    """
    reset_log(filepath)
    
    msg1 = (
        "Finished trial {0}/{1}, ".format(trial_number+1, num_trials) +
        "elapsed={}, ".format(elapsed)
    )
    msg2 = (
        "n={0}, d={1}, eta={2}, M={3}, ".format(n,d,eta,M) +
        "r={0}, c={1}, ".format(r,c) +
        "data={0}, rows={1}, columns={2}".format(data_filepath, rows, columns)
    )

    msg = msg1 + msg2
    
    logging.info(msg)
    print(msg1)
    print("Recorded log to {}".format(filepath))
    return None

def mnist_logger(filepath, num, eta, F1, elapsed):
    """
    After a trial has been completed, logs details to the specified file

    Input:
        filepath (str): log file
        num (int): which number of MNIST has been benchmarked
        eta (float): % inliers
        F1 (float): F1 score
        elapsed (float): elapsed time

    Return:
        None
    """
    reset_log(filepath)

    msg = "num={0}, eta={1}, F1={2}, elapsed={3}".format(num, eta, F1, elapsed)
    logging.info(msg)
    print(msg)
    print("Recorded log to {}".format(filepath))
    return None

def get_times_from_log(filepath, calc_avg=True) -> list:
    """
    Scans the given log file and returns a list of the runtimes

    Input:
        filepath (str): filepath of the log file to be scanned
        calc_avg (bool): if False then returns list containing lists for each trial, if True then returns list of average times
    
    Output:
        times (list): list of runtimes
    """
    times = []
    trial_times = []

    with open(filepath, "r") as f:
        #num_trials = f.readline().split(sep=", ")[0].split("/")[1]

        for line in f: # iterate over each line
            line_split = line.split(sep=", ") # split into elements of a list
            first_part = line_split[0]

            if "Finished trial" in first_part: # indicator that this is a log written by benchmark_logger
                num_trials = first_part.split("/")[1]
                time = float(line_split[1].split("=")[1]) # line_split[1] will be "elapsed=<time>"
                trial_times.append(time)

                trial_num = first_part.split()[-1].split("/")[0] # which trial number we are on
                if trial_num == num_trials: # i.e. if we are on trial 5/5
                    times.append(trial_times)
                    trial_times = [] # reset

    if calc_avg:
        times = calc_avg_times(times)
    
    return times

def get_M_from_log(filepath):
    """
    Scans given log file and returns list of M parameters used

    Input:
        filepath (str): filepath of the log file to be scanned
    
    Return:
        M_list (list): list of M values found
    """
    M_list = []
    with open(filepath, "r") as f:
        for line in f:
            line_split = line.split(sep=", ")
            if "Finished trial" in line_split[0]: # indicator that this is a log written by benchmark_logger
                M_part = line_split[5]
                M_val = float(M_part.split("=")[-1])
                if M_val not in M_list:
                    M_list.append(M_val)

    return M_list

def get_r_from_log(filepath):
    num_trials = 5
    r_list = []
    r_trials = []
    with open(filepath, "r") as f:
        for line in f:
            line_split = line.split(sep=", ")
            first_part = line_split[0]
            if "Finished trial" in first_part: # indicator that this is a log written by benchmark_logger
                r_part = line_split[6]
                r_val = float(r_part.split("=")[-1])
                r_trials.append(r_val)

                trial_num = int(first_part.split()[-1].split("/")[0]) # which trial number we are on
                if trial_num == num_trials: # i.e. if we are on trial 5/5
                    r_avg = np.mean(r_trials)
                    r_list.append(r_avg)
                    r_trials = [] # reset

    return r_list


def timeout(log_Path, time_limit=20):
    print("Received no input within {} seconds. Continuing.".format(time_limit))
    log_Path.unlink()
    return None

def check_log(log_file, time_limit=20) -> None:
    """
    Checks if the given log_file already exists, and if so asks the user if it should continue.

    Input:
        log_file (str): file path for log file
        time_limit (float): time limit for user prompt
    
    Return:
        None
    """
    if log_file is not None:
        log_Path = Path(log_file)
        if log_Path.exists():
            try:
                print("Log file {0} already exists. Overwriting in {1}s. Press Ctrl+C to abort.".format(log_file, time_limit))
                time.sleep(time_limit)
            except KeyboardInterrupt:
                exit("Aborting.")
    
    return None

def notify() -> None:
    """
    Sends a notification to the notify-run channel in notify.sh
    """
    subprocess.call(["sh", "src/notify.sh"])
    return None

def format_eta(eta) -> str:
    """
    Formats eta for use in a file name, e.g. 0.95 -> 0p95

    Input:
        eta (float): eta value
    
    Return:
        eta_str (str): eta as a string
    """
    eta_str = str(eta).replace(".","p")
    return eta_str

def find_trial_param(params):
    """
    From a dictionary of parameters, identifies which is a list and returns the name of
    that parameter and the list
    """
    trial_param = None
    # find which parameter is the list
    for param, val in params.items():
        if type(val) == list:
            trial_param = param
            trial_param_vals = val.copy()
            break
    
    if trial_param is None:
        raise TypeError("No list of parameters was found")
    
    return trial_param, trial_param_vals

def confirm_benchmark(types):
    run = {typ: False for typ in types}
    for typ in types:
        msg = input("Run benchmarks for {}? (y/n) ".format(typ))
        if msg == "y":
            run[typ] = True
    
    confirmed_types = {typ: types[typ] for typ in types if run[typ] == True}
    return confirmed_types

def get_title(func_name, n, d, eta):
    """
    Formats a title for plotting

    Input:
        func_name (str): name of method being benchmarked
        n, d, eta (list/float): parameters, one of which is a list
    
    Return:
        title (str): formatted title
    """
    # find key param being benchmarked
    params = {"n": n, "d": d, "eta": eta}
    for param in params:
        if type(params[param]) == list:
            key_param = param
            break
    
    # remove key param from params dictionary
    params = {param: params[param] for param in params if param != key_param}

    # initialise method names
    func_names = {
        "relaxation_heuristic": "Relaxation-Based Heuristic",
        "shrink": "Shrink Heuristic",
        "shrink_avg": "Shrink (avg) Heuristic",
        "shenmaier": "Shenmaier's Approximation",
        "dcmeb": "DCMEB"
    }

    # first part of title
    title = "Runtime for {0} as a function of {1}, ".format(func_names[func_name], key_param)

    # add each param value
    for param in params:
        title += "{0}={1}, ".format(param, params[param])
    
    # remove ", " from end of title
    title = title[:-2]
    return title