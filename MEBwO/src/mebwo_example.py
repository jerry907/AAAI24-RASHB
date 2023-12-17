import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('C:/Users/28626/Documents/FedML_rfa/Minimum-Enclosing-Balls-with-Outliers/src')

import data.generation
from meb.ball import MEBwO
from meb.geometry import M_estimate
import plot_settings


def main():
    n = 1000
    d = 2

    # mod = importlib.import_module(model_path)
    # ClientModel = getattr(mod, 'ClientModel')

    np.random.seed(2000)
    gdata = data.generation.normal(0, 1, n, d)
    M = M_estimate(gdata)
    ball = MEBwO().fit(data=gdata, method="shenmaier", eta=0.9, M=M, outputflag=1)

    ball.plot(gdata, show=False)
    plt.legend()
    plt.savefig("../images/sample_mebwo.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
