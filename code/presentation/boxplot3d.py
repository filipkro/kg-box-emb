from matplotlib.gridspec import GridSpec
from functools import partial
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle, Patch
import plotly.graph_objects as go
import numpy as np
from pprint import pprint

TEST_BOXES_MINDELTA = {
    "class_1": np.array(
        [
            [[0.5, 0.5, 0.5], [1.9, 1.9, 1.9]],
            [[0.4, 0.4, 0.4], [1.7, 1.7, 1.7]],
            [[0.3, 0.3, 0.3], [1.5, 1.5, 1.5]],
            [[0.2, 0.2, 0.2], [1.3, 1.3, 1.3]],
            [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]],
        ]
    ),
    "class_2": np.array(
        [
            [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]],
            [[0.2, 0.3, 0.3], [1.0, 1.2, 1.2]],
            [[0.2, 0.4, 0.4], [1.0, 1.4, 1.4]],
            [[0.2, 0.5, 0.5], [1.0, 1.6, 1.6]],
            [[0.2, 0.6, 0.6], [1.0, 1.8, 1.8]],
        ]
    ),
}


def softplus(z):
    return np.log(1 + np.exp(z))
    # return np.log(1 + np.exp(-np.abs(z))) + np.max(z, 0)


def box_to_vertices(w, d, mindelta=False):
    if mindelta:
        x, y, z = w
        X, Y, Z = w + softplus(d)
    else:
        x, y, z = w
        X, Y, Z = w + d
        # X, Y, Z = w[0] + d[0], w[1] + d[1], w[2] + d[2]
    return [
        [(x, y, z), (X, y, z), (X, Y, z), (x, Y, z)],
        [(x, y, z), (X, y, z), (X, y, Z), (x, y, Z)],
        [(x, y, z), (x, Y, z), (x, Y, Z), (x, y, Z)],
        [(x, y, Z), (X, y, Z), (X, Y, Z), (x, Y, Z)],
        [(x, Y, z), (X, Y, z), (X, Y, Z), (x, Y, Z)],
        [(X, y, z), (X, Y, z), (X, Y, Z), (X, y, Z)],
    ]


def plot_min_delta_boxes_3d_matplotlib(
    w_list,
    d_list,
    colors=None,
    color_legend=None,
    alphas=None,
    linewidths=None,
    draw_labels=False,
    labels=None,
    title="Box Embeddings",
    fig=None,
    ax=None,
):
    """
    Plot multiple boxes given lists of lower left corners (w_list) and widths/heights (d_list) using matplotlib.
    Optionally provide a list of labels to display at the box centers.
    """
    from matplotlib.patches import Rectangle

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    if labels is None:
        labels = [None] * len(w_list)
    if colors is None:
        colors = [None] * len(w_list)
    if alphas is None:
        alphas = [1] * len(w_list)
    if linewidths is None:
        linewidths = map(lambda a: 2 * a, alphas)

    for w, d, label, color, alpha, lw in zip(
        w_list, d_list, labels, colors, alphas, linewidths
    ):
        if color is None:
            color = "black"
        ax.add_collection3d(
            Poly3DCollection(
                box_to_vertices(w, d),
                alpha=alpha / 3,
                edgecolor=color,
                linewidth=lw / 10,
                facecolor=color,
                zorder=alpha * 10,
            )
        )
        # if draw_labels and label:
        #     label_x = upper_right[0] + 0.02 * (upper_right[0] - lower_left[0])
        #     label_y = upper_right[1]
        #     ax.text(
        #         label_x,
        #         label_y,
        #         label,
        #         ha="left",
        #         va="center",
        #         fontsize=10,
        #         color="black",
        #         bbox=dict(
        #             facecolor="white",
        #             edgecolor=color,
        #             boxstyle="round,pad=0.2",
        #             linewidth=1.5,
        #         ),
        #         zorder=11,
        #     )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.grid(True)

    # Optionally, auto-scale axes to fit all boxes
    all_x = [w[0] for w in w_list] + [
        w[0] + softplus(d[0]) for w, d in zip(w_list, d_list)
    ]
    all_y = [w[1] for w in w_list] + [
        w[1] + softplus(d[1]) for w, d in zip(w_list, d_list)
    ]
    ax.set_xlim(
        min(all_x) - 0.10 * (max(all_x) - min(all_x)),
        max(all_x) + 0.30 * (max(all_x) - min(all_x)),
        # -1.5,
        # 3.0,
    )
    ax.set_ylim(
        min(all_y) - 0.10 * (max(all_y) - min(all_y)),
        max(all_y) + 0.30 * (max(all_y) - min(all_y)),
        # -1.8,
        # 8.0,
    )

    if color_legend is not None:
        plt.legend(
            handles=[Patch(color=k, label=v) for k, v in color_legend.items()],
            loc="upper right",
        ).set_zorder(99)

    return fig, ax
