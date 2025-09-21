# %%
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


################################################################################
# legend
################################################################################
def ax_plot_legend(
    color_dict,
    marker="s",
    marker_size=10,
    marker_edgecolor="black",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    text_size=10,
    ax=None,
):
    """
    Plot a legend for a given color dictionary on a specified axes object.

    Parameters
    ----------
    color_dict : dict
        A dictionary where keys are labels and values are colors (hex codes).
    marker : str, optional
        The marker style to use in the legend. Default is "s" (square).
    marker_size : int, optional
        The size of the markers in the legend. Default is 10.
    marker_edgecolor : str, optional
        The edge color of the markers in the legend. Default is "black".
    loc : str, optional
        The location of the legend. Default is "center left".
    bbox_to_anchor : tuple, optional
        The bounding box anchor for the legend. Default is (1.05, 0.5).
    text_size : int, optional
        The font size of the legend text. Default is 10.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot the legend on. If None, uses the current axes.
        Default is None.
    """
    if ax is None:
        ax = plt.gca()

    # Create legend handles with custom markers for each label
    handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=marker,
            markerfacecolor=color_dict[label],
            markeredgecolor=marker_edgecolor,
            markersize=marker_size,
            label=label,
            linestyle="None",  # No line, only marker
        )
        for label in color_dict.keys()
    ]

    # Plot the legend with specified parameters
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=text_size,
    )


################################################################################
# scale bar
################################################################################
def ax_plot_rgb_with_scalebar(
    img_rgb,
    color_rgb=(255, 255, 255),
    mpp=0.50,
    bar_width_um=100,
    bar_height_px=10,
    bar_anchor=(0.05, 0.95),
    text_size=12,
    text_to_bar_px=20,
    text_unit="μm",
    text_weight="normal",
    ax=None,
):
    """
    Plot an RGB image with an overlaid scale bar.

    Parameters
    ----------
    img_rgb : numpy.ndarray
        The RGB image array with shape (height, width, 3).
    color_rgb : tuple, optional
        RGB color values for the scale bar. Default is (255, 255, 255) (white).
    mpp : float, optional
        Microns per pixel ratio for the image. Default is 0.50.
    bar_width_um : int, optional
        Width of the scale bar in micrometers. Default is 100.
    bar_height_px : int, optional
        Height of the scale bar in pixels. Default is 10.
    bar_anchor : tuple, optional
        Relative position (x, y) for scale bar placement. Default is (0.05, 0.95).
    text_size : int, optional
        Font size for the scale bar text. Default is 12.
    text_to_bar_px : int, optional
        Distance in pixels between the scale bar and text label. Default is 20.
    text_unit : str, optional
        Unit symbol to display with the scale bar. Default is "μm".
    text_weight : str, optional
        Font weight for the text. Default is "normal".
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, uses the current axes. Default is None.
    """
    if ax is None:
        ax = plt.gca()

    # Get image dimensions
    img_h, img_w, _ = img_rgb.shape

    # Calculate scale bar dimensions in pixels
    width_px = int(bar_width_um / mpp)  # Convert micrometers to pixels

    # Calculate scale bar position based on anchor point
    xmin = int(img_w * bar_anchor[0])
    xmax = xmin + width_px

    # Center the bar vertically around the anchor y-position
    ymin = int(img_h * bar_anchor[1] - bar_height_px / 2)
    ymax = ymin + bar_height_px

    # Create a copy of the image and draw the scale bar
    img_plot = img_rgb.copy().astype(np.uint8)
    img_plot[ymin:ymax, xmin:xmax, :] = color_rgb

    # Calculate text position (to the right of the scale bar)
    x_text = xmax + text_to_bar_px
    y_text = int((ymin + ymax) / 2)  # Center text vertically with the bar

    # Format the text label
    text = f"{bar_width_um} {text_unit}"

    # Add text annotation
    ax.text(
        x_text,
        y_text,
        text,
        color=np.array(color_rgb) / 255,
        fontsize=text_size,
        fontweight=text_weight,
        ha="left",
        va="center",
    )

    ax.imshow(img_plot)


################################################################################
# test
################################################################################
if __name__ == "__main__":
    import importlib

    import matplotlib.pyplot as plt
    import numpy as np

    from pycodex.visualization import utils

    # Reload module for development
    importlib.reload(utils)
    ax_plot_rgb_with_scalebar = utils.ax_plot_rgb_with_scalebar
    ax_plot_legend = utils.ax_plot_legend

    img = np.zeros((1000, 1000, 3))
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax_plot_legend({"A": "#FF5733", "B": "#33FF57", "C": "#3357FF"})
    ax_plot_rgb_with_scalebar(
        img,
        mpp=0.50,
        bar_width_um=100,
        text_to_bar_px=10,
        text_size=10,
    )

# %%
