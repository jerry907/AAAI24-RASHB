import pandas as pd

import benchmarking.utils

heuristics = ["relaxation_heuristic", "shenmaier", "shrink", "shrink_avg"]
data_types = ["hyperspherical_shell", "normal", "uniform_ball", "uniform_ball_with_outliers"]
params = {
    "n": [1000+3000*i for i in range(10)],
    "d": [10+10*i for i in range(10)],
    "eta": [0.5 + 0.1*i for i in range(5)]
}

fixed_params = {
    "n": "d30",
    "d": "n10000",
    "eta": "n10000_d30"
}

param_latex = {
    "n": "$n$",
    "d": "$d$",
    "eta": "$\\eta$"
}

column_headers = {
    "relaxation_heuristic": "RBH",
    "shenmaier": "Shenm.",
    "shrink": "Shrink",
    "shrink_avg": "APH"
}

for data_type in data_types:
    for param in params:
        results_dict = {heuristic: None for heuristic in heuristics}
        times_dict = {heuristic: None for heuristic in heuristics}
        
        for heuristic in heuristics:
            # construct filename
            filename = "func_{0}_{1}".format(param, fixed_params[param])
            if param != "eta":
                filename += "_eta0p9"
            filename += "_{}".format(data_type)
            
            filepath = r"benchmarks/{0}/{1}/{2}.log".format(heuristic, data_type, filename)
            results = benchmarking.utils.get_r_from_log(filepath)
            times = benchmarking.utils.get_times_from_log(filepath)

            for lst in [results, times]:
                while len(lst) < len(params[param]):
                    lst.append(0)
            results_dict[heuristic] = results
            times_dict[heuristic] = times
        
        table_dict = {param_latex[param]: params[param]}
        for heuristic in heuristics:
            header = column_headers[heuristic]
            table_dict["{} $r$".format(header)] = results_dict[heuristic]
            table_dict["{} $t$".format(header)] = times_dict[heuristic]

        results_df = pd.DataFrame.from_dict(table_dict)
        results_df = results_df.round(decimals=2)

        results_df.to_csv(r"tables/{0}/{1}.csv".format(data_type, filename), index=False, sep="&")
            