import matplotlib.pyplot as plt

def scatter_2d(data, color, alpha, label):
    """
    Creates a scatter plot for 2d data

    Input:
        data (array like): data to be plotted (MUST BE 2D)
        color (str): colour of plotted data
        alpha (float): transparency
        label (str): label for plot legend
    
    Return:
        None
    """
    n = len(data)
    x_data = [data[i][0] for i in range(n)]
    y_data = [data[i][1] for i in range(n)]
    plt.scatter(x_data, y_data, color=color, alpha=alpha, label=label)

    return None