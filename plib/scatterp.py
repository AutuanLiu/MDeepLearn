import matplotlib.pyplot as plt


def scatterp(x_data, y_data, x_label='', y_label='', title='',
             yscale_log=False):
    """
    散点图对于展示两个变量之间的关系非常有用;
    可以添加另一个参数，如数据点的半径来编码第三个变量，从而可视化三个变量之间的关系
    """
    _, ax = plt.subplots()
    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s=10, color=color, alpha=0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
