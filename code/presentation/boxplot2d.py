from matplotlib.gridspec import GridSpec
from functools import partial
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import numpy as np
from pprint import pprint

FPS = 25
KEY_FRAME_INTERVAL = 40  # ms per key frame
FRAME_LENGTH = KEY_FRAME_INTERVAL / FPS
TEST_BOXES = {
    "class_1": np.array(
        [
            [[0.5, 0.5], [1.9, 1.9]],
            [[0.4, 0.4], [1.7, 1.7]],
            [[0.3, 0.3], [1.5, 1.5]],
            [[0.2, 0.2], [1.3, 1.3]],
            [[0.1, 0.1], [1.1, 1.1]],
        ]
    ),
    "class_2": np.array(
        [
            [[0.2, 0.2], [1.0, 1.0]],
            [[0.2, 0.3], [1.0, 1.2]],
            [[0.2, 0.4], [1.0, 1.4]],
            [[0.2, 0.5], [1.0, 1.6]],
            [[0.2, 0.6], [1.0, 1.8]],
        ]
    ),
}
TEST_BOXES_MINDELTA = {
    "class_1": np.array(
        [
            [[0.5, 0.5], [1.9, 1.9]],
            [[0.4, 0.4], [1.7, 1.7]],
            [[0.3, 0.3], [1.5, 1.5]],
            [[0.2, 0.2], [1.3, 1.3]],
            [[0.1, 0.1], [1.1, 1.1]],
        ]
    ),
    "class_2": np.array(
        [
            [[0.2, 0.2], [1.0, 1.0]],
            [[0.2, 0.3], [1.0, 1.2]],
            [[0.2, 0.4], [1.0, 1.4]],
            [[0.2, 0.5], [1.0, 1.6]],
            [[0.2, 0.6], [1.0, 1.8]],
        ]
    ),
}
# TEST_LOSSES = np.vstack([
#     np.arange(TEST_BOXES["class_1"].shape[0]),
#     np.random.rand(4, TEST_BOXES["class_1"].shape[0])
# ]).T
TEST_LOSSES = np.random.rand(TEST_BOXES["class_1"].shape[0], 4)


def softplus(z):
    return np.log(1 + np.exp(z))


def plot_box_2d(z, Z, fig=None, ax=None):
    """
    Plot a box given lower left and upper right corners.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        returnFig = True
    else:
        returnFig = False

    # ax.set_xlim(
    #     np.floor(z[0]-0.5),
    #     np.ceil(Z[0]+0.5)
    # )
    # ax.set_ylim(
    #     np.floor(z[1]-0.5),
    #     np.ceil(Z[1]+0.5)
    # )
    ax.set_aspect("equal")
    ax.set_title("2D Box Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    lower_left = z
    upper_right = Z
    width = upper_right[0] - lower_left[0]
    height = upper_right[1] - lower_left[1]
    rect = Rectangle(
        lower_left, width, height, fill=None, edgecolor="blue", linewidth=1
    )
    ax.add_patch(rect)

    if returnFig:
        return fig, ax


def plot_min_delta_box_2d(w, d, fig=None, ax=None):
    """
    Plot a box given lower left and upper right corners.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        returnFig = True
    else:
        returnFig = False

    # ax.set_xlim(
    #     np.floor(z[0]-0.5),
    #     np.ceil(Z[0]+0.5)
    # )
    # ax.set_ylim(
    #     np.floor(z[1]-0.5),
    #     np.ceil(Z[1]+0.5)
    # )
    ax.set_aspect("equal")
    ax.set_title("2D Box Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    lower_left = w
    upper_right = w + softplus(d)
    width = upper_right[0] - lower_left[0]
    height = upper_right[1] - lower_left[1]
    rect = Rectangle(
        lower_left, width, height, fill=None, edgecolor="blue", linewidth=1
    )
    ax.add_patch(rect)

    if returnFig:
        return fig, ax


def plot_min_delta_boxes_2d_matplotlib(
    w_list,
    d_list,
    colors=None,
    alphas=None,
    draw_labels=False,
    labels=None,
    fig=None,
    ax=None,
):
    """
    Plot multiple boxes given lists of lower left corners (w_list) and widths/heights (d_list) using matplotlib.
    Optionally provide a list of labels to display at the box centers.
    """
    from matplotlib.patches import Rectangle

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = [None] * len(w_list)
    if colors is None:
        colors = [None] * len(w_list)
    if alphas is None:
        alphas = [1] * len(w_list)

    for w, d, label, color, alpha in zip(w_list, d_list, labels, colors, alphas):
        lower_left = w
        upper_right = w + softplus(d)
        width = upper_right[0] - lower_left[0]
        height = upper_right[1] - lower_left[1]
        if color is None:
            color = "black"
        rect = Rectangle(
            lower_left,
            width,
            height,
            fill=False,
            edgecolor=color,
            alpha=alpha,
            linewidth=alpha * 2,
            zorder=alpha * 10,
        )
        ax.add_patch(rect)
        if draw_labels and label:
            ax.text(
                lower_left[0] + width / 2,
                lower_left[1] + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                zorder=alpha * 10,
            )

    ax.set_aspect("equal")
    ax.set_title("2D Box Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)

    # Optionally, auto-scale axes to fit all boxes
    all_x = [w[0] for w in w_list] + [
        w[0] + softplus(d[0]) for w, d in zip(w_list, d_list)
    ]
    all_y = [w[1] for w in w_list] + [
        w[1] + softplus(d[1]) for w, d in zip(w_list, d_list)
    ]
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
    ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)

    return fig, ax


def plot_min_delta_boxes_2d_plotly(
    w_list,
    d_list,
    colors=None,
    draw_labels=False,
    labels=None,
    fig=None,
    edge_points=50,
):
    """
    Plot multiple boxes given lists of lower left corners (w_list) and widths/heights (d_list) using Plotly.
    Optionally provide a list of labels for hover.
    """
    if fig is None:
        fig = go.Figure()
    if labels is None:
        labels = [None] * len(w_list)
    if colors is None:
        colors = [None] * len(w_list)

    for w, d, label, color in zip(w_list, d_list, labels, colors):
        lower_left = w
        upper_right = w + softplus(d)
        x0, y0 = lower_left
        x1, y1 = upper_right

        if color is None:
            color = "black"

        # Draw the rectangle as a shape
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color=color, width=1),
            fillcolor="rgba(0,0,0,0)",
            name=label,
        )

        if draw_labels:
            # Densely sample points along the perimeter
            xs = []
            ys = []
            # Bottom edge
            xs += list(np.linspace(x0, x1, edge_points))
            ys += [y0] * edge_points
            # Right edge
            xs += [x1] * edge_points
            ys += list(np.linspace(y0, y1, edge_points))
            # Top edge
            xs += list(np.linspace(x1, x0, edge_points))
            ys += [y1] * edge_points
            # Left edge
            xs += [x0] * edge_points
            ys += list(np.linspace(y1, y0, edge_points))

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=8, opacity=0),
                    hoverinfo="text",
                    text=[label] * len(xs) if label else ["" for _ in range(len(xs))],
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="2D Box Plot",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def interpolate_boxes(boxes, frames=10):
    zxp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    zyp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    Zxp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    Zyp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)

    zXp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 0, 0])
    zYp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 0, 1])
    ZXp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 1, 0])
    ZYp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 1, 1])

    return np.array([[zXp, ZXp], [zYp, ZYp]]).T.reshape(-1, 2, 2)


def interpolate_losses(losses, frames=10):
    xp = np.linspace(0, len(losses) - 1, frames, endpoint=True)

    tlp = np.interp(xp, np.arange(len(losses)), losses[:, 0])
    plp = np.interp(xp, np.arange(len(losses)), losses[:, 1])
    nlp = np.interp(xp, np.arange(len(losses)), losses[:, 2])
    rlp = np.interp(xp, np.arange(len(losses)), losses[:, 3])

    return np.array([[xp, tlp, plp, nlp, rlp]]).T.reshape(-1, 5)


def calculate_intersection(box1, box2):
    """
    Calculate the intersection of two boxes.
    Each box is defined by its lower left and upper right corners.
    Returns the bottom left and top right vertices.

    TODO: add mindelta functionality
    """
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[1][0], box2[1][0])
    y2 = min(box1[1][1], box2[1][1])

    if x2 < x1 or y2 < y1:
        return None  # No intersection

    return np.array([[x1, y1], [x2, y2]])


def animate_boxes(
    boxes,
    losses=None,
    save=False,
    fp="training.mp4",
):
    """
    Animate several series of boxes, each stored in a dictionary with a key.
    Smooth transitions between boxes.
    Each box is defined by its lower left and upper right corners.
    """
    if losses is not None:
        print(f"{next(iter(boxes.values())).shape[0]} ?= {len(losses)}")
        assert next(iter(boxes.values())).shape[0] == len(losses)
    duration = (
        next(iter(boxes.values())).shape[0] * KEY_FRAME_INTERVAL / 1000.0
    )  # in seconds
    frames = int(duration * FPS) + 1
    print(f"Duration: {duration}")
    print(f"Frames:   {frames}")

    boxes_interpolated = {
        key: interpolate_boxes(series, frames=frames) for key, series in boxes.items()
    }

    if losses is not None:
        losses_interpolated = interpolate_losses(losses, frames=frames)

    fig, axs = plt.subplots(2, 2)
    ((ax, ax2), (ax3, ax4)) = axs
    gs = ax3.get_gridspec()
    ax3.remove()
    ax4.remove()
    axloss = fig.add_subplot(gs[-1, :])

    # fig = plt.figure()

    # Set limits based on minimum and maximum values
    ax.set_xlim(
        np.floor(
            np.min([np.min(series[:, 0, 0]) for series in boxes_interpolated.values()])
            - 0.5
        ),
        np.ceil(
            np.max([np.max(series[:, 1, 0]) for series in boxes_interpolated.values()])
            + 0.5
        ),
    )
    ax.set_ylim(
        np.floor(
            np.min([np.min(series[:, 0, 1]) for series in boxes_interpolated.values()])
            - 0.5
        ),
        np.ceil(
            np.max([np.max(series[:, 1, 1]) for series in boxes_interpolated.values()])
            + 0.5
        ),
    )
    ax.set_aspect("equal")
    ax.set_title("Box embeddings during training")
    ax.set_xlabel("Embed dim 1")
    ax.set_ylabel("Embed dim 2")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(False)

    ax2.axis("off")
    epoch_text_label = ax2.text(0, 0.9, f"EPOCH:")
    loss_text_label = ax2.text(0, 0.8, f"LOSS:")
    pos_loss_text_label = ax2.text(0, 0.7, f"POS RATIO:")
    neg_loss_text_label = ax2.text(0, 0.6, f"NEG RATIO:")
    reg_loss_text_label = ax2.text(0, 0.5, f"REG LOSS:")
    epoch_text = ax2.text(0.5, 0.9, f"")
    loss_text = ax2.text(0.5, 0.8, f"")
    pos_loss_text = ax2.text(0.5, 0.7, f"")
    neg_loss_text = ax2.text(0.5, 0.6, f"")
    reg_loss_text = ax2.text(0.5, 0.5, f"")

    axloss.set_title("Loss Plot")
    axloss.set_xlabel("Epoch")
    axloss.set_ylabel("Loss")
    axloss.set_xlim(0, np.max(losses_interpolated[:, 0]))
    axloss.set_ylim(
        np.min(losses_interpolated[:1, 1:]), np.max(losses_interpolated[:1, 1:])
    )
    axloss.axhline(y=0, color="black", linewidth=1)
    axloss.grid(True)
    if losses is not None:
        tot_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 1], label="Total Loss"
        )[0]
        pos_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 2], label="Pos. Ratio"
        )[0]
        neg_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 3], label="Neg. Ratio"
        )[0]
        reg_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 4], label="Reg. Loss"
        )[0]
        axloss.legend()

    rects = {}

    # Draw rectangles
    for i, key in enumerate(boxes_interpolated.keys()):
        rects[key] = Rectangle(
            (0, 0),
            0,
            0,
            fill=None,
            linewidth=2 if len(boxes_interpolated) < 20 else 0.4,
            alpha=1 if len(boxes_interpolated) < 20 else 0.7,
        )
        ax.add_patch(rects[key])

    def update(frame, losses=losses, mindelta=True):
        for key, rect in rects.items():
            boxes_series = boxes_interpolated[key]
            lower_left = boxes_series[frame][0]
            if mindelta:
                upper_right = boxes_series[frame][0] + softplus(boxes_series[frame][1])
            else:
                upper_right = boxes_series[frame][1]
            width = upper_right[0] - lower_left[0]
            height = upper_right[1] - lower_left[1]
            rect.set_xy(lower_left)
            rect.set_width(width)
            rect.set_height(height)
            try:
                XMIN = min(XMIN, lower_left[0], upper_right[0])
                XMAX = max(XMAX, lower_left[0], upper_right[0])
                YMIN = min(YMIN, lower_left[1], upper_right[1])
                YMAX = max(YMAX, lower_left[1], upper_right[1])
            except NameError:
                XMIN = min(lower_left[0], upper_right[0])
                XMAX = max(lower_left[0], upper_right[0])
                YMIN = min(lower_left[1], upper_right[1])
                YMAX = max(lower_left[1], upper_right[1])

        # Set limits
        XYMIN = min(XMIN, YMIN)
        XYMAX = max(XMAX, YMAX)

        ax.set_xlim(XYMIN - 0.2 * abs(XYMIN), XYMAX + 0.2 * abs(XYMAX))
        ax.set_ylim(XYMIN - 0.2 * abs(XYMIN), XYMAX + 0.2 * abs(XYMAX))

        del XMIN, XMAX, YMIN, YMAX, XYMIN, XYMAX

        # Update Text
        epoch_text.set_text(f"{int(np.floor(losses_interpolated[frame, 0]))}")
        loss_text.set_text(f"{losses_interpolated[frame, 1]:10.5f}")
        pos_loss_text.set_text(f"{losses_interpolated[frame, 2]:10.5f}")
        neg_loss_text.set_text(f"{losses_interpolated[frame, 3]:10.5f}")
        reg_loss_text.set_text(f"{losses_interpolated[frame, 4]:10.5f}")

        if losses is not None:
            # Update losslines
            tot_loss.set_xdata(losses_interpolated[:frame, 0])
            tot_loss.set_ydata(losses_interpolated[:frame, 1])

            pos_loss.set_xdata(losses_interpolated[:frame, 0])
            pos_loss.set_ydata(losses_interpolated[:frame, 2])

            neg_loss.set_xdata(losses_interpolated[:frame, 0])
            neg_loss.set_ydata(losses_interpolated[:frame, 3])

            reg_loss.set_xdata(losses_interpolated[:frame, 0])
            reg_loss.set_ydata(losses_interpolated[:frame, 4])

            if frame > 0:
                axloss.set_ylim(
                    np.min(losses_interpolated[:frame, 1:]),
                    np.max(losses_interpolated[:frame, 1:]),
                )

        return (
            ax,
            rects.values(),
            epoch_text,
            loss_text,
            pos_loss_text,
            neg_loss_text,
            reg_loss_text,
            axloss,
            tot_loss,
            pos_loss,
            neg_loss,
            reg_loss,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=FRAME_LENGTH, repeat=True, blit=False
    )
    fig.tight_layout()
    if save:
        ani.save("training.mp4", fps=FPS)
    else:
        plt.show()


def animate_boxes_with_blitting(
    boxes,
    losses,
    save=False,
    fp="training.mp4",
    box_filter=lambda k: True,
    box_filter_type="omit",
    box_labels=None,
    box_label_filter=lambda k: True,
):
    """
    Animate several series of boxes, each stored in a dictionary with a key.
    Smooth transitions between boxes.
    Each box is defined by its lower left and upper right corners.
    """
    if box_labels is None:
        box_labels = {}
    else:
        box_labels = {
            k: v.split("/")[-1].split("#")[-1] if box_label_filter(k) else ""
            for k, v in box_labels.items()
        }

    filtered_boxes = {k: v for k, v in boxes.items() if box_filter(k)}
    if box_filter_type == "omit":
        boxes = filtered_boxes
        box_labels = {k: v for k, v in box_labels.items() if box_filter(k)}

    print([l for k, l in box_labels.items() if l != ""])

    if losses is not None:
        print(f"{next(iter(boxes.values())).shape[0]} ?= {len(losses)}")
        assert next(iter(boxes.values())).shape[0] == len(losses)
    duration = (
        next(iter(boxes.values())).shape[0] * KEY_FRAME_INTERVAL / 1000.0
    )  # in seconds
    frames = int(duration * FPS) + 1
    print(f"Duration: {duration}")
    print(f"Frames:   {frames}")

    boxes_interpolated = {
        key: interpolate_boxes(series, frames=frames) for key, series in boxes.items()
    }

    if losses is not None:
        losses_interpolated = interpolate_losses(losses, frames=frames)
        # print(losses_interpolated)

    AnimationArtists = namedtuple(
        "AnimationArtists",
        [
            "epoch_text",
            "total_loss_text",
            "pos_loss_text",
            "neg_loss_text",
            "reg_loss_text",
            "total_loss",
            "pos_loss",
            "neg_loss",
            "reg_loss",
        ]
        + [f"box_{key}" for key in boxes.keys()]
        + [f"box_{key}_label" for key in boxes.keys()],
    )
    print(f"`boxes`s has {len(boxes.keys())} keys.")
    print(f"`boxes_interpolated` has {len(boxes_interpolated.keys())} keys.")
    print(f"`box_labels` has {len(box_labels.keys())} keys.")

    FixedArtists = namedtuple(
        "FixedArtists",
        ("ax", "ax2", "ax_loss_total", "ax_loss_pos", "ax_loss_neg", "ax_loss_reg"),
    )

    def init_fig(fig, fixed_artists):

        fixed_artists.ax.set_aspect("equal")
        fixed_artists.ax.set_title("Box embeddings during training")
        fixed_artists.ax.set_xlabel("Embed dim 1")
        fixed_artists.ax.set_ylabel("Embed dim 2")
        fixed_artists.ax.set_xlim(-0.5, 0.5)
        fixed_artists.ax.set_ylim(-0.5, 0.5)
        fixed_artists.ax.grid(False)
        fixed_artists.ax.get_xaxis().set_visible(False)
        fixed_artists.ax.get_yaxis().set_visible(False)

        fixed_artists.ax2.axis("off")
        epoch_text_label = fixed_artists.ax2.text(-0.4, 0.9, f"EPOCH:")
        loss_text_label = fixed_artists.ax2.text(-0.4, 0.7, f"LOSS:")
        pos_loss_text_label = fixed_artists.ax2.text(-0.4, 0.5, f"POS RATIO:")
        neg_loss_text_label = fixed_artists.ax2.text(-0.4, 0.3, f"NEG RATIO:")
        reg_loss_text_label = fixed_artists.ax2.text(-0.4, 0.1, f"REG LOSS:")

        # axloss.set_title("Loss Plot")
        fixed_artists.ax_loss_reg.set_xlabel("Epoch")
        fixed_artists.ax_loss_total.set_ylabel("Total Loss")
        fixed_artists.ax_loss_pos.set_ylabel("Pos. Loss")
        fixed_artists.ax_loss_neg.set_ylabel("Neg. Loss")
        fixed_artists.ax_loss_reg.set_ylabel("Reg. Loss")

        for i, axis in enumerate(fixed_artists[-4:]):
            axis.set_xlim(0, np.max(losses_interpolated[:, 0]))
            # print(
            #     f"Loss Minimum: {np.min(losses_interpolated[:1, i + 1])}\nLoss Maximum: {np.max(losses_interpolated[:1, i + 1])}")
            axis.set_ylim(
                np.min([0, np.min(losses_interpolated[:, i + 1])]),
                np.max([1, np.max(losses_interpolated[:, i + 1])]),
            )
            axis.axhline(y=0, color="black", linewidth=1)
            axis.grid(True)
            if i < 3:
                # axis.get_xaxis().set_visible(False)
                axis.get_xaxis().set_tick_params(labelbottom=False)

        animation_artists = AnimationArtists(
            fixed_artists.ax2.text(0.2, 0.9, f""),
            fixed_artists.ax2.text(0.2, 0.7, f""),
            fixed_artists.ax2.text(0.2, 0.5, f""),
            fixed_artists.ax2.text(0.2, 0.3, f""),
            fixed_artists.ax2.text(0.2, 0.1, f""),
            fixed_artists.ax_loss_total.plot(
                losses_interpolated[0, 0], losses_interpolated[0, 1], label="Total Loss"
            )[0],
            fixed_artists.ax_loss_pos.plot(
                losses_interpolated[0, 0], losses_interpolated[0, 2], label="Pos. Ratio"
            )[0],
            fixed_artists.ax_loss_neg.plot(
                losses_interpolated[0, 0], losses_interpolated[0, 3], label="Neg. Ratio"
            )[0],
            fixed_artists.ax_loss_reg.plot(
                losses_interpolated[0, 0], losses_interpolated[0, 4], label="Reg. Loss"
            )[0],
            *[
                fixed_artists.ax.add_patch(
                    Rectangle(
                        (0, 0),
                        0,
                        0,
                        fill=None,
                        color=(
                            "red"
                            if (box_filter(key) and box_filter_type == "bold")
                            else "black"
                        ),
                        linewidth=(
                            2
                            if (box_filter(key) and box_filter_type == "bold")
                            else 0.8 if len(boxes_interpolated) < 20 else 0.2
                        ),
                        alpha=(
                            1
                            if len(boxes_interpolated) < 20
                            or (box_filter(key) and box_filter_type == "bold")
                            else 0.7
                        ),
                        zorder=1.1 if box_filter(key) else 1,
                    )
                )
                for key in boxes_interpolated.keys()
            ]
            + [
                fixed_artists.ax.text(
                    0, 0, label, ha="center", va="center", fontsize=10, color="red"
                )
                for label in box_labels.values()
            ],
        )

        return animation_artists

    def update_artists(
        frame, animation_artists, boxes_interpolated, losses_interpolated, mindelta=True
    ):
        for key, rect, label in zip(
            boxes_interpolated.keys(),
            animation_artists[9 : 9 + len(boxes_interpolated.keys())],
            animation_artists[9 + len(boxes_interpolated.keys()) :],
        ):
            boxes_series = boxes_interpolated[key]
            lower_left = boxes_series[frame][0]
            if mindelta:
                upper_right = boxes_series[frame][0] + softplus(boxes_series[frame][1])
            else:
                upper_right = boxes_series[frame][1]
            width = upper_right[0] - lower_left[0]
            height = upper_right[1] - lower_left[1]
            rect.set_xy(lower_left)
            rect.set_width(width)
            rect.set_height(height)

            # Update box label positions
            if label.get_text() != "":
                label.set_position(rect.get_center())

            try:
                XMIN = min(XMIN, lower_left[0], upper_right[0])
                XMAX = max(XMAX, lower_left[0], upper_right[0])
                YMIN = min(YMIN, lower_left[1], upper_right[1])
                YMAX = max(YMAX, lower_left[1], upper_right[1])
            except NameError:
                XMIN = min(lower_left[0], upper_right[0])
                XMAX = max(lower_left[0], upper_right[0])
                YMIN = min(lower_left[1], upper_right[1])
                YMAX = max(lower_left[1], upper_right[1])

        # Set limits
        XYMIN = min(XMIN, YMIN)
        XYMAX = max(XMAX, YMAX)

        fixed_artists.ax.set_xlim(XYMIN - 0.3 * abs(XYMIN), XYMAX + 0.3 * abs(XYMAX))
        fixed_artists.ax.set_ylim(XYMIN - 0.3 * abs(XYMIN), XYMAX + 0.3 * abs(XYMAX))

        del XMIN, XMAX, YMIN, YMAX, XYMIN, XYMAX

        zoom_offset = 50
        if frame > zoom_offset:
            fixed_artists.ax_loss_pos.set_ylim(
                0, max(losses_interpolated[max(0, frame - zoom_offset) : frame, 2])
            )
            fixed_artists.ax_loss_neg.set_ylim(
                0, max(losses_interpolated[max(0, frame - zoom_offset) : frame, 3])
            )

        fixed_artists.ax.set_aspect("equal")

        # Update Text
        animation_artists.epoch_text.set_text(
            f"{int(np.floor(losses_interpolated[frame, 0]))}"
        )
        animation_artists.total_loss_text.set_text(
            f"{losses_interpolated[frame, 1]:10.5f}"
        )
        animation_artists.pos_loss_text.set_text(
            f"{losses_interpolated[frame, 2]:10.5f}"
        )
        animation_artists.neg_loss_text.set_text(
            f"{losses_interpolated[frame, 3]:10.5f}"
        )
        animation_artists.reg_loss_text.set_text(
            f"{losses_interpolated[frame, 4]:10.5f}"
        )

        if losses is not None:
            # Update losslines
            animation_artists.total_loss.set_xdata(losses_interpolated[:frame, 0])
            animation_artists.total_loss.set_ydata(losses_interpolated[:frame, 1])

            animation_artists.pos_loss.set_xdata(losses_interpolated[:frame, 0])
            animation_artists.pos_loss.set_ydata(losses_interpolated[:frame, 2])

            animation_artists.neg_loss.set_xdata(losses_interpolated[:frame, 0])
            animation_artists.neg_loss.set_ydata(losses_interpolated[:frame, 3])

            animation_artists.reg_loss.set_xdata(losses_interpolated[:frame, 0])
            animation_artists.reg_loss.set_ydata(losses_interpolated[:frame, 4])

            # if frame > 0:
            #     axloss.set_ylim(np.min(losses_interpolated[:frame, 1:]), np.max(
            #         losses_interpolated[:frame, 1:]))

        return animation_artists

    # 1. Create the plot
    fig = plt.figure(layout="constrained")
    gs = GridSpec(5, 7, figure=fig, hspace=0.1)

    fixed_artists = FixedArtists(
        fig.add_subplot(gs[:, 2:]),
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[1, :2]),
        fig.add_subplot(gs[2, :2]),
        fig.add_subplot(gs[3, :2]),
        fig.add_subplot(gs[4, :2]),
    )

    animation_artists = init_fig(fig, fixed_artists)

    # 3. Apply the three plotting functions written above
    # init = partial(init_fig, fig=fig, fixed_artists=fixed_artists)
    # step = partial(frame_iter)
    update = partial(
        update_artists,
        animation_artists=animation_artists,
        boxes_interpolated=boxes_interpolated,
        losses_interpolated=losses_interpolated,
    )

    # 4. Generate the animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=frames,
        interval=FRAME_LENGTH,
        repeat=True,
        blit=True,
    )
    fig.tight_layout()
    # # 5. Save the animation
    # anim.save(
    #     filename='test_blit.mp4',
    #     fps=FPS,
    #     extra_args=['-vcodec', 'libx264'],
    #     dpi=300,
    # )
    # # plt.show()
    if save:
        anim.save(fp, fps=FPS)
    else:
        plt.show()


if __name__ == "__main__":

    w = np.array([0.2, 0.2])
    d = np.array([0.4, 0.6])

    plot_min_delta_box_2d(w, d)
    plt.show()

    # Animate boxes
    # animate_boxes(TEST_BOXES, losses=TEST_LOSSES, save=True)
    # plt.show()
    # animate_boxes_with_blitting(TEST_BOXES, losses=TEST_LOSSES)
    # animate_boxes_with_blitting(TEST_BOXES_MINDELTA, losses=TEST_LOSSES)
