import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#using-the-helper-function-code-style
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1])) #, labels=col_labels)
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0])) #, labels=row_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, thdata=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     textweights=("regular", "regular"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    thdata
        Data used to determine threshold.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    textweights
        A pair of font weights.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    
    if not isinstance(thdata, (list, np.ndarray)):
        thdata = data.copy()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(thdata[i, j]) > threshold)])
            #kw.update(weight=textweights[int(im.norm(thdata[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__=="__main__":
    import pandas as pd
    from pathlib import Path
    import argparse
    plt.rcParams['font.size'] = 16

    datadir = Path('data')
    figdir = Path('fig/res')

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--sample",type=str,default="all",\
        help="sample (all, near, far)")
    argsin = parser.parse_args()
    sample = argsin.sample

    # load csv data
    aspect = pd.read_csv(datadir/f'{sample}_mul-mul_aspect.csv',index_col=['FT','member'])
    slope  = pd.read_csv(datadir/f'{sample}_mul-mul_slope.csv',index_col=['FT','member'])

    nmethod = aspect.columns.size
    for n in range(nmethod):
        key = aspect.columns[n]
        print(key)
        figdir1 = figdir / key.lower()
        rows = aspect.index.get_level_values('FT')
        rows = rows[~rows.duplicated()]
        cols = aspect.index.get_level_values('member')
        cols = cols[~cols.duplicated()]
        print(rows,len(rows))
        print(cols,len(cols))
        adata = aspect[key].values.reshape(len(rows),len(cols))
        sdata = slope[key].values.reshape(len(rows),len(cols))
        print(adata)
        row_labels = [f'FT{f}' for f in rows]
        col_labels = [f'mem{m}' for m in cols]
        fig1, ax1 = plt.subplots(figsize=[6,4],constrained_layout=True)
        fig2, ax2 = plt.subplots(figsize=[6,4],constrained_layout=True)
        im1, cbar1 = heatmap(adata, row_labels, col_labels, ax=ax1, \
            cbarlabel='aspect ratio', cbar_kw={'pad':0.01},\
            vmin=1.0,vmax=4.0)
        im2, cbar2 = heatmap(sdata, row_labels, col_labels, ax=ax2, \
            cbarlabel='slope (degree)', cbar_kw={'pad':0.01},\
            vmin=30.0,vmax=60.0,cmap='PiYG')
        _ = annotate_heatmap(im1,textcolors=("white", "black"))
        _ = annotate_heatmap(im2,thdata=np.abs(sdata-45.0),threshold=10.0,\
            textcolors=("black", "white"),\
            valfmt='{x:.0f}')
        ax1.set_title(key.lower()+f' {sample} multi-multi')
        ax2.set_title(key.lower()+f' {sample} multi-multi')
        fig1.savefig(figdir1/f'{sample}_mul-mul_aspect.png',dpi=300)
        fig2.savefig(figdir1/f'{sample}_mul-mul_slope.png',dpi=300)
        plt.show()
        plt.close()
        #exit()