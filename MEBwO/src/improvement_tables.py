import pandas as pd

heuristics = ["dcmeb", "dcssh"]
data_types = ["normal", "hyperspherical_shell", "uniform_ball", "uniform_ball_with_outliers"]
params = ["n", "d"]

fixed_params = {"n": "d100", "d": "n1000"}

def update_header(header):
    if header == "n" or header == "d":
        new_header = "${}$".format(header)
    elif header == "avg%":
        new_header = "Avg\\%"
    elif "^" in header:
        new_header = "$\\hat{{{0}}}_{1}$".format(header[0], header[1])
    else:
        new_header = "${}$".format(header[0] + "_" + header[1])

    return new_header

for heuristic in heuristics:
    for data_type in data_types:
        for param in params:
            filepath = "benchmarks/{0}/{1}/avg_pct_func_{2}_{3}_eta0p9_{4}".format(heuristic, data_type, param, fixed_params[param], data_type)
            data = pd.read_csv("{}.csv".format(filepath))
            
            old_headers = list(data)
            new_headers = {old_header: update_header(old_header) for old_header in old_headers}

            data = data.rename(columns=new_headers)
            data = data.round(decimals=2)

            data.to_csv("{}_table.csv".format(filepath), index=False, sep="&")
