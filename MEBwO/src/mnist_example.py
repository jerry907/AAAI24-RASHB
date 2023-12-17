from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"datasets/mnist_train.csv", header=None)

numbers = range(10)
num_examples = 5
numbers_dfs = [df[df[0] == number].head(num_examples).drop(labels=0, axis=1) for number in numbers]

fig, axs = plt.subplots(num_examples,10)

for i in range(num_examples):
    for number in numbers:
        img = np.array(numbers_dfs[number].iloc[i])
        pixels = img.reshape((28,28))

        ax = axs[i][number]

        ax.imshow(pixels, cmap="binary")

        # remove ticks from axes
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False
        )

        # remove borders
        ax.axis("off")

plt.savefig(r"images/mnist_example.png", bbox_inches="tight")