import matplotlib.pyplot as plt
import os

import pandas as pd


def plot_with_zoom(df, test_idx, save_fig=False, directory=None, name=None, **cols_and_labels):
    fig, ax = plt.subplots(figsize=(20, 8))
    for key, value in cols_and_labels.items():
        ax.plot(df[key], label=value)

    if df.index[0] < pd.to_datetime('2009-01-01'):
        # Where insert should be placed and how big should be (proportionally to the image)
        axins = ax.inset_axes([0.6, 0.05, .3, .3])
        for key in cols_and_labels.keys():
            axins.plot(df.loc[test_idx, key])

        axins.set_xticklabels([])
        axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins, edgecolor="black", label='zoom')
    ax.legend()

    if save_fig:
        plt.savefig(os.path.join(directory, f'{name}.png'))
    plt.show()