import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import data.generation
import plot_settings

n = 1000

np.random.seed(956)

datas = {
    "normal_2d": data.generation.normal(0,1, n, 2),
    "normal_3d": data.generation.normal(0,1, n, 3),
    "uniform_ball_2d": data.generation.uniform_ball(n, 2, 1),
    "uniform_ball_3d": data.generation.uniform_ball(n, 3, 1),
    "hyperspherical_shell_2d": data.generation.hyperspherical_shell(n, 2, 1, 2),
    "hyperspherical_shell_3d": data.generation.hyperspherical_shell(n, 3, 1, 2),
    "uniform_ball_with_outliers_2d": data.generation.uniform_ball_with_ouliters(n, 2, 0.9, 1, 2, 3),
    "uniform_ball_with_outliers_3d": data.generation.uniform_ball_with_ouliters(n, 3, 0.9, 1, 2, 3)
}



for name, data in datas.items():
    dim = name.split("_")[-1]
    if name.startswith("hyperspherical_shell"):
        c = [np.linalg.norm(x-np.zeros(int(dim[0]))) for x in data]
        cmap = "crest"
    else:
        c = None
        cmap = None

    fig = plt.figure(figsize=(6,6))
    if dim == "2d":
        plt.scatter(data[:,0], data[:,1], alpha=0.5, c=c, cmap=cmap)
        plt.gca().set_aspect("equal")
    
    if dim == "3d":
        
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        ax.scatter(data[:,0], data[:,1], data[:,2], alpha=0.5, c=c, cmap=cmap)
    
    if dim == "2d":
        plt.savefig(r"images/data_plots/{0}".format(name), bbox_inches='tight')
    else:
        plt.savefig(r"images/data_plots/{0}".format(name), bbox_inches='tight', transparent=True)

    plt.clf()
