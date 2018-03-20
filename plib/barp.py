"""
当对类别数很少（<10）的分类数据进行可视化时，条形图是最有效的。当类别数太多时，
条形图将变得很杂乱，难以理解。你可以基于条形的数量观察不同类别之间的区别，不同的类别可以轻易地分离以及用颜色分组
"""
import matplotlib.pyplot as plt
import numpy as np


def barp(x_data, y_data, error_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color='#539caf', align='center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    # 误差条形是额外添加在每个条形中心上的线，可用于表示标准差。
    ax.errorbar(x_data, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


def stackedbarp(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    """
    堆叠条形图非常适合于可视化不同变量的分类构成。
    Parameters
    ----------
    y_data_list: 一组列表，其中每个子列表代表了一个不同的组。然后我们循环地遍历每一个组，
    并在 X 轴上绘制柱体和对应的值，每一个分组的不同类别将使用不同的颜色表示。
    """
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color=colors[i], align='center', label=y_data_names[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(
                x_data,
                y_data_list[i],
                color=colors[i],
                bottom=y_data_list[i - 1],
                align='center',
                label=y_data_names[i])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')


def groupedbarp(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    """
    分组条形图允许我们比较多个类别变量。
    Parameters
    ----------
    y_data_list: 一组列表，其中每个子列表代表了一个不同的组。然后我们循环地遍历每一个组，
    并在 X 轴上绘制柱体和对应的值，每一个分组的不同类别将使用不同的颜色表示。
    """
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width / 2), total_width / 2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(x_data + alteration[i], y_data_list[i], color=colors[i], label=y_data_names[i], width=ind_width)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')
