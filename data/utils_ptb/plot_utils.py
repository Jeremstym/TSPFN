import pandas as pd
import torch
import numpy as np
from typing import Union
import matplotlib
import matplotlib.pyplot as plt


def plot_ecg(points: Union[np.array, torch.Tensor, list],
             c: Union[str, list] = 'blue',
             title: str = '',
             labels: Union[str, list]=None, size: int = 20,
             location_legend: str = 'best'):

    font = {
        # 'family': 'normal',
        # 'weight': 'bold',
        'size': size
    }

    matplotlib.rc('font', **font)

    plt.figure(figsize=(10, 4), dpi=300)
    if isinstance(points, list):
        t = np.arange(len(points[0]))
        for i in range(len(points)):
            p = points[i]
            if isinstance(c, list):
                plt.plot(t, p, color=c[i], lw=.7, label=labels[i])
            else:
                plt.plot(t, p, lw=.7, label=labels[i])

        plt.plot(np.zeros(len(p)), color='black', lw=.2)
    else:
        t = np.arange(len(points))
        plt.plot(t, points, color=c, lw=.7, label=labels)
        plt.plot(np.zeros(len(points)), color='black', lw=.2)

    plt.title('ECG Signal')
    plt.xlabel('Timepoints')
    plt.ylabel('Amplitude')
    plt.legend(loc=location_legend)
    plt.title(title)
    plt.grid(False)
    plt.show()
    return plt


def adjust_seaborn_plot(plot,
                        width=10,
                        height=4,
                        dpi=300,
                        legend=True,
                        legend_loc='best',
                        font_scale=1.0,
                        xlabel=None,
                        ylabel=None):
    import seaborn as sns
    """
    Adjusts the dimensions, DPI, legend, size, xlabel, and ylabel of a Seaborn plot.

    Parameters:
    - plot: Seaborn plot object.
    - width: Width of the plot in inches.
    - height: Height of the plot in inches.
    - dpi: Dots per inch (resolution) of the plot.
    - legend: Boolean, whether to display the legend.
    - legend_loc: Location of the legend (e.g., 'best', 'upper right', 'lower left', etc.).
    - font_scale: Font scale factor for adjusting text size on the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.

    Returns:
    - None
    """
    # Access the figure associated with the plot's Axes
    fig = plot.get_figure()
    fig.set_figwidth(width)
    fig.set_figheight(height)
    fig.set_dpi(dpi)

    if legend:
        plot.legend(loc=legend_loc)
    sns.set(font_scale=font_scale)

    if xlabel is not None:
        plot.set_xlabel(xlabel)
    if ylabel is not None:
        plot.set_ylabel(ylabel)


def spectogram3d(y, srate=500, ax=None, title=None):
    from matplotlib import mlab
    fig = plt.figure(dpi=300)
    if not ax:
        ax = plt.axes(projection='3d')
    ax.set_title(title, loc='center', wrap=True)
    spec, freqs, t = mlab.specgram(y, Fs=srate)
    X, Y, Z = t[None, :], freqs[:, None],  20.0 * np.log10(spec)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequencies (Hz)')
    ax.set_zlabel('amplitude (dB)')
    # ax.set_zlim(-140, 0)
    return X, Y, Z

