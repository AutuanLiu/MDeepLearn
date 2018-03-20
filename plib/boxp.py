import matplotlib.pyplot as plt


def boxp(x_data, y_data, base_color="#539caf", median_color="#297083", x_label="", y_label="", title=""):
    """
    实线箱的底部表示第一个四分位数，顶部表示第三个四分位数，箱内的线表示第二个四分位数（中位数）
    Parameters
    ----------
    x_data: 变量的列表
    y_data: y_data 的每一列或 y_data 序列中的每个向量绘制一个箱线图，因此 x_data 中的每个值
    对应 y_data 中的一列/一个向量。
    """
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    # patch_artist must be True to control box fill
    # Properties of median line
    #
    # Properties of box
    # Properties of whiskers
    # Properties of whisker caps
    ax.boxplot(
        y_data,
        patch_artist=True,
        medianprops={'color': median_color},
        boxprops={
            'color': base_color,
            'facecolor': base_color
        },
        whiskerprops={'color': base_color},
        capprops={'color': base_color})

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
