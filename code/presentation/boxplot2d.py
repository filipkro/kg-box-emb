from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import numpy as np

FPS = 25
KEY_FRAME_INTERVAL = 75  # ms per key frame
FRAME_LENGTH = KEY_FRAME_INTERVAL / FPS
TEST_BOXES = {
    "class_1": np.array([
        [
            [0.5, 0.5],
            [1.9, 1.9]
        ],
        [
            [0.4, 0.4],
            [1.7, 1.7]
        ],
        [
            [0.3, 0.3],
            [1.5, 1.5]
        ],
        [
            [0.2, 0.2],
            [1.3, 1.3]
        ],
        [
            [0.1, 0.1],
            [1.1, 1.1]
        ],
    ]),
    "class_2": np.array([
        [
            [0.2, 0.2],
            [1.0, 1.0]
        ],
        [
            [0.2, 0.3],
            [1.0, 1.2]
        ],
        [
            [0.2, 0.4],
            [1.0, 1.4]
        ],
        [
            [0.2, 0.5],
            [1.0, 1.6]
        ],
        [
            [0.2, 0.6],
            [1.0, 1.8]
        ],
    ]),
}


def plot_box_2d(z, Z, fig=None, ax=None):
    '''
    Plot a box given lower left and upper right corners.
    '''
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
    ax.set_aspect('equal')
    ax.set_title('2D Box Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)
    lower_left = z
    upper_right = Z
    width = upper_right[0] - lower_left[0]
    height = upper_right[1] - lower_left[1]
    rect = Rectangle(lower_left, width, height,
                     fill=None, edgecolor='blue', linewidth=1)
    ax.add_patch(rect)

    if returnFig:
        return fig, ax


def interpolate_boxes(boxes, frames=10):
    zxp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    zyp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    Zxp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)
    Zyp = np.linspace(0, len(boxes) - 1, frames, endpoint=True)

    zXp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 0, 0])
    zYp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 0, 1])
    ZXp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 1, 0])
    ZYp = np.interp(zxp, np.arange(len(boxes)), boxes[:, 1, 1])

    return np.array([
        [zXp, ZXp],
        [zYp, ZYp]
    ]).T.reshape(-1, 2, 2)


def interpolate_losses(losses, frames=10):
    xp = np.linspace(0, len(losses) - 1, frames, endpoint=True)

    tlp = np.interp(xp, np.arange(len(losses)), losses[:, 0])
    plp = np.interp(xp, np.arange(len(losses)), losses[:, 1])
    nlp = np.interp(xp, np.arange(len(losses)), losses[:, 2])
    rlp = np.interp(xp, np.arange(len(losses)), losses[:, 3])

    return np.array([
        [xp, tlp, plp, nlp, rlp]
    ]).T.reshape(-1, 5)


def calculate_intersection(box1, box2):
    '''
    Calculate the intersection of two boxes.
    Each box is defined by its lower left and upper right corners.
    Returns the bottom left and top right vertices.
    '''
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[1][0], box2[1][0])
    y2 = min(box1[1][1], box2[1][1])

    if x2 < x1 or y2 < y1:
        return None  # No intersection

    return np.array([[x1, y1], [x2, y2]])


def animate_boxes(boxes, losses=None, save=False):
    '''
    Animate several series of boxes, each stored in a dictionary with a key.
    Smooth transitions between boxes.
    Each box is defined by its lower left and upper right corners.
    '''
    if losses is not None:
        print(f"{next(iter(boxes.values())).shape[0]} ?= {len(losses)}")
        assert next(iter(boxes.values())).shape[0] == len(losses)
    duration = next(iter(boxes.values())
                    ).shape[0] * KEY_FRAME_INTERVAL / 1000.0  # in seconds
    frames = int(duration * FPS) + 1
    print(f"Duration: {duration}")
    print(f"Frames:   {frames}")

    boxes_interpolated = {
        key: interpolate_boxes(series, frames=frames)
        for key, series in boxes.items()
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
        np.floor(np.min([np.min(series[:, 0, 0])
                         for series in boxes_interpolated.values()])-0.5),
        np.ceil(np.max([np.max(series[:, 1, 0])
                        for series in boxes_interpolated.values()])+0.5)
    )
    ax.set_ylim(
        np.floor(np.min([np.min(series[:, 0, 1])
                         for series in boxes_interpolated.values()])-0.5),
        np.ceil(np.max([np.max(series[:, 1, 1])
                        for series in boxes_interpolated.values()])+0.5)
    )
    ax.set_aspect('equal')
    ax.set_title('Box embeddings during training')
    ax.set_xlabel('Embed dim 1')
    ax.set_ylabel('Embed dim 2')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(False)

    ax2.axis('off')
    epoch_text_label = ax2.text(0, 0.9,    f"EPOCH:")
    loss_text_label = ax2.text(0, 0.8,     f"LOSS:")
    pos_loss_text_label = ax2.text(0, 0.7, f"POS RATIO:")
    neg_loss_text_label = ax2.text(0, 0.6, f"NEG RATIO:")
    reg_loss_text_label = ax2.text(0, 0.5, f"REG LOSS:")
    epoch_text = ax2.text(0.4, 0.9,    f"")
    loss_text = ax2.text(0.4, 0.8,     f"")
    pos_loss_text = ax2.text(0.4, 0.7, f"")
    neg_loss_text = ax2.text(0.4, 0.6, f"")
    reg_loss_text = ax2.text(0.4, 0.5, f"")

    axloss.set_title("Loss Plot")
    axloss.set_xlabel("Epoch")
    axloss.set_ylabel("Loss")
    axloss.set_xlim(0, np.max(losses_interpolated[:, 0]))
    axloss.set_ylim(np.min(losses_interpolated[:1, 1:]), np.max(
        losses_interpolated[:1, 1:]))
    axloss.axhline(y=0, color='black', linewidth=1)
    axloss.grid(True)
    if losses is not None:
        tot_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 1], label='Total Loss')[0]
        pos_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 2], label='Pos. Ratio')[0]
        neg_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 3], label='Neg. Ratio')[0]
        reg_loss = axloss.plot(
            losses_interpolated[0, 0], losses_interpolated[0, 4], label='Reg. Loss')[0]
        axloss.legend()

    rects = {}

    # Draw rectangles
    for i, key in enumerate(boxes_interpolated.keys()):
        rects[key] = Rectangle(
            (0, 0), 0, 0, fill=None,
            linewidth=2 if len(boxes_interpolated) < 20 else 0.4,
            alpha=1 if len(boxes_interpolated) < 20 else 0.7
        )
        ax.add_patch(rects[key])

    def update(frame, losses=losses):
        for key, rect in rects.items():
            boxes_series = boxes_interpolated[key]
            lower_left = boxes_series[frame][0]
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
        loss_text.set_text(f"{losses_interpolated[frame, 1]:.10f}")
        pos_loss_text.set_text(f"{losses_interpolated[frame, 2]:.10f}")
        neg_loss_text.set_text(f"{losses_interpolated[frame, 3]:.10f}")
        reg_loss_text.set_text(f"{losses_interpolated[frame, 4]:.10f}")

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
                axloss.set_ylim(np.min(losses_interpolated[:frame, 1:]), np.max(
                    losses_interpolated[:frame, 1:]))

        return ax, rects.values(), epoch_text, loss_text, pos_loss_text, neg_loss_text, reg_loss_text, axloss, tot_loss, pos_loss, neg_loss, reg_loss

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=FRAME_LENGTH, repeat=True, blit=False)
    fig.tight_layout()
    if save:
        ani.save("training.mp4", fps=FPS)
    else:
        plt.show()


if __name__ == "__main__":

    # Animate boxes
    animate_boxes(TEST_BOXES)
