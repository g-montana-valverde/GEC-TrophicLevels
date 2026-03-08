"""
Utility functions to plot *paired* train / test confusion matrices, with an
optional common colour‑bar that does **not** alter the size of the matrices.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay


# --------------------------------------------------------------------------- #
# helper: a single confusion matrix without an internal colour-bar
# --------------------------------------------------------------------------- #
def plot_single_cm(conf_matrix,
                   labels,
                   ax,
                   *,
                   vmin,
                   vmax,
                   cmap="Blues",
                   fontsize=16,
                   **plot_kwargs):
    """
    Internal helper – plots on *ax* and returns the display object.

    Parameters
    ----------
    conf_matrix : array-like
        Confusion matrix values.
    labels : list-like
        Display labels.
    ax : Matplotlib Axes
        Axes to plot into.
    vmin, vmax : float
        Colour scale limits.
    cmap : str or Colormap, default "Blues"
        Colormap for the matrix.
    fontsize : int, default 16
        Font size for the numbers inside the confusion matrix.
    **plot_kwargs :
        Additional keyword arguments forwarded to ConfusionMatrixDisplay.plot.
    """
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot(ax=ax,
              cmap=cmap,
              colorbar=False,  # we manage colour-bar outside
              **plot_kwargs)

    # Enforce shared colour scale
    disp.im_.set_clim(vmin=vmin, vmax=vmax)

    # Make the numbers larger
    if hasattr(disp, "text_") and disp.text_ is not None:
        for txt in disp.text_.ravel():
            txt.set_fontsize(fontsize)

    return disp


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def plot_train_test_confusion_matrix(cm_train,
                                     cm_test,
                                     labels,
                                     *,
                                     cmap=None,
                                     show_colorbar=True,
                                     bar_width=0.05,
                                     **plot_kwargs):
    """
    Side‑by‑side confusion matrices with an *optional* shared colour‑bar.

    Parameters
    ----------
    cm_train, cm_test : array-like (n_classes, n_classes)
        Confusion matrices.
    labels            : list-like
        Class names (same order as the matrices).
    cmap              : Matplotlib colormap or str
    show_colorbar     : bool, default True
        Whether to draw a colour‑bar in its own narrow Axes.
    bar_width         : float, fraction of figure width reserved for the bar.
    **plot_kwargs     : forwarded to `ConfusionMatrixDisplay.plot`
                        (values_format, xticks_rotation, …)

    Returns
    -------
    fig : Matplotlib Figure
    """
    # common colour limits
    vmin = 0
    vmax = 1

    # --- figure & layout ----------------------------------------------------
    if show_colorbar:
        # 3 columns: train | test | colour‑bar
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 3,
                      width_ratios=[1, 1, bar_width / (1 - 2*bar_width)],
                      wspace=0.20,
                      figure=fig)

        ax_train = fig.add_subplot(gs[0, 0])
        ax_test  = fig.add_subplot(gs[0, 1])
        cax      = fig.add_subplot(gs[0, 2])
    else:
        # only the two matrices
        fig, (ax_train, ax_test) = plt.subplots(1, 2,
                                                figsize=(10, 5))
        cax = None

    # --- plot the matrices --------------------------------------------------
    disp_train = plot_single_cm(cm_train, labels, ax_train, vmin=vmin, vmax=vmax,
                                cmap=cmap, **plot_kwargs)
    ax_train.set_title("Train Confusion Matrix")

    disp_test  = plot_single_cm(cm_test, labels, ax_test, vmin=vmin, vmax=vmax,
                                cmap=cmap, **plot_kwargs)
    ax_test.set(title="Test Confusion Matrix", ylabel=None)
    ax_test.axes.yaxis.set_ticklabels([])

    # --- optional colour‑bar ------------------------------------------------
    if show_colorbar:
        fig.colorbar(disp_test.im_, cax=cax)
        cax.set_ylabel("Counts")

    plt.tight_layout()
    return fig
