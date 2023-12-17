import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plot_settings

heuristics = ["shenmaier", "shrink", "shrink_avg"]
eta_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

column_headers = {
    "shenmaier": "Shenmaier",
    "shrink": "Shrink Heuristic",
    "shrink_avg": "APH"
}

times_dict = {heuristic: None for heuristic in heuristics}
F1_dict = {"$\eta$": eta_list}
F1_dict["ABOD"] = [0.950, 0.896, 0.841, 0.781, 0.717, 0.648]
F1_dict["OCSVM"] = [0.965, 0.934, 0.903, 0.868, 0.833, 0.797]
F1_dict["DRAE"] = [0.974, 0.949, 0.925, 0.898, 0.867, 0.838]
F1_dict["HDMY"] = [0.965, 0.935, 0.907, 0.883, 0.857, 0.831]

for heuristic in heuristics:
    filepath = r"benchmarks/{}/mnist/mnist_log.log".format(heuristic)
    time = 0
    scores = []
    F1 = [None]*10
    with open(filepath, "r") as f:
        for line in f:
            split = line.split(", ")

            time += float(split[3].split("=")[-1])
            
            num = int(split[0][-1])
            F1[num] = float(split[2].split("=")[-1])

            if num == 9:
                scores.append(np.mean(F1))
                F1 = [None]*10
    
    times_dict[heuristic] = time

    scores = list(reversed([round(x,3) for x in scores]))
    F1_dict[column_headers[heuristic]] = scores



F1_df = pd.DataFrame.from_dict(F1_dict)

fig, ax = plt.subplots()

colours = sns.color_palette("hls", 7)
markers = [".", "v", "^", "s", "x", "D", "p"]
i = 0
for column in F1_df:
    if column != "$\eta$":
        plt.plot(eta_list, F1_df[column], label=column, marker=markers[i], c=colours[i])
        i += 1

ax.set_xlim(0.95, 0.7)
plt.xlabel("$\eta$")
plt.ylabel("$F_1$ Score")
plt.legend()
plt.savefig(r"images/mnist_plot.png", bbox_inches="tight")

F1_df.to_csv("tables/mnist_table.csv", index=False, sep="&")
print(times_dict)
