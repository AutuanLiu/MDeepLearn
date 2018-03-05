import matplotlib.pyplot as plt


def linep(x_data, y_data, x_label="", y_label="", title=""):
    """
    当一个变量随另一个变量的变化而变化的幅度很大时，即它们有很高的协方差时，线图非常好用
    """
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
