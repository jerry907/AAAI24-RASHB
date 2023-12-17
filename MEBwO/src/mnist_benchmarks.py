import pandas as pd
import numpy as np

import benchmarking.trials, benchmarking.utils
from meb.mebwo_algorithms import alg__relaxation_heuristic, alg__shrink, alg__shrink_avg, alg__shenmaier

np.random.seed(10000)

numbers = range(10)
eta_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

funcs = {
    "shenmaier": alg__shenmaier,
    "shrink_avg": alg__shrink_avg,
    "relaxation_heuristic": alg__relaxation_heuristic,
    "shrink": alg__shrink
}
print("Run for:")
i = 1
for func in funcs:
    print("\t{} {}?".format(i, func))
    i += 1
msg = int(input()) - 1

func = list(funcs.values())[msg]
print(func)
log_file = "benchmarks/"
log_file += list(funcs.keys())[msg]
log_file += "/mnist/mnist_log.log"

df = pd.read_csv(r"datasets/mnist_train.csv", header=None)
print("Finished loading data")
for eta in eta_list:
    for number in numbers:
        benchmarking.trials.mnist_benchmark(df=df, func=func, number=number, eta=eta, log_file=log_file)

benchmarking.utils.notify()