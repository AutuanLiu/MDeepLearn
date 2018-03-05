import matplotlib.pyplot as plt


def histp(data, n_bins, cumulative=False, x_label="", y_label="", title=""):
    """
    直方图对于观察或真正了解数据点的分布十分有用

    Parameters
    ----------
    n_bins: 控制直方图的箱体数量或离散化程度。更多的箱体或柱体能给我们提供更多的信息，
    但同样也会引入噪声并使我们观察到的全局分布图像变得不太规则。而更少的箱体将给我们更多的全局信息，
    我们可以在缺少细节信息的情况下观察到整体分布的形状。
    cumulative: 布尔值，它允许我们选择直方图是不是累积的，即选择概率密度函数（PDF）或累积密度函数（CDF）
    """
    _, ax = plt.subplots()
    ax.hist(data, n_bins=n_bins, cumulative=cumulative, color='#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
