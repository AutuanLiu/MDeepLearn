import matplotlib.pyplot as plt
import numpy as np


# Overlay 2 histograms to compare them
def overhistp(data1,
              data2,
              n_bins=0,
              data1_name="",
              data1_color="#539caf",
              data2_name="",
              data2_color="#7663b0",
              x_label="",
              y_label="",
              title=""):
    """
    首先，我们设定的水平区间要同时满足两个变量的分布。根据水平区间的范围和箱体数，我们可以计算
    每个箱体的宽度。其次，我们在一个图表上绘制两个直方图，需要保证一个直方图存在更大的透明度。
    """
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    bin_width = (data_range[1] - data_range[0]) / max_nbins

    if n_bins == 0:
        bins = np.arange(data_range[0], data_range[1] + bin_width, bin_width)
    else:
        bins = n_bins

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins=bins, color=data1_color, alpha=1, label=data1_name)
    ax.hist(data2, bins=bins, color=data2_color, alpha=0.75, label=data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='best')
