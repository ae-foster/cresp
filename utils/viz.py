# visualisation helpers for data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def scatter_plot(points, labels=None):
    # plt.clf()
    fig, ax = plt.subplots()
    if labels is None:
        plt.scatter(points[:, 0], points[:, 1], s=5)
    elif len(labels.unique()) < 20:
        unique_labels = np.unique(labels)
        colours = sns.color_palette("husl", len(unique_labels))
        for label in unique_labels:
            idx = np.where(labels == label)
            plt.scatter(
                points[idx][:, 0],
                points[idx][:, 1],
                color=colours[label],
                label=label,
                s=5,
                alpha=0.4,
                edgecolors="none",
            )
        ax.legend()
    else:
        cmap = sns.color_palette("husl", as_cmap=True)
        if labels.shape[1] == 1:
            labels, s = labels, 5
        else:
            s = 5 / labels[:, 1].mean() * labels[:, 1]
            labels = labels[:, 0]
        sc = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, s=s, alpha=0.4, edgecolors="none")
        plt.colorbar(sc)
    return plt


def gp_plot(ax, x, y, X, Y, y_pred, sigma, title=""):
    fontsize = 30
    colours = sns.color_palette("Paired")
    ax.plot(x, y, "r:", lw=4, alpha=0.9, label=r"$f(x)$", color=colours[5])
    ax.plot(X, Y, "r.", markersize=17, label="Observations", color=colours[5])
    if y_pred is not None:
        ax.plot(x, y_pred, "b-", lw=4, alpha=0.9, label="Prediction", color=colours[1])
    ax.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=0.8,
        fc=colours[0],
        ec="None",
        label="95% confidence interval",
    )
    ax.set_title(title)
    plt.xlabel("$\\xi$", fontsize=fontsize)
    plt.ylabel("$x$", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=30)
    return plt