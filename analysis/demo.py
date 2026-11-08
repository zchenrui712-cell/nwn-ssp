"""Run the portfolio workflow with synthetic readouts and export local plots."""

import argparse
from hashlib import file_digest
from pathlib import Path

import matplotlib
import numpy as np

from workflow import (
    centroid_distances, choose_images, class_statistics, collect_features,
    encode_pixels, load_mnist_csv, pca_view, silhouette_comparison, sine_input,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def synthetic_images():
    """Generate four groups of simple 28-by-28 patterns with repeatable noise."""
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(1, 5), 25)
    images = rng.normal(0.08, 0.04, (labels.size, 28, 28))
    for image, digit in zip(images, labels):
        column = 4 + 4 * digit + rng.integers(-2, 3)
        image[4:24, column:column + 3] += 0.8
        image[5 * digit:5 * digit + 2, 5:22] += 0.4
    return np.clip(images, 0, 1).reshape(-1, 784), labels


def synthetic_readouts(stimulus):
    """Generate four illustrative channels from each input independently."""
    previous = np.r_[stimulus[0], stimulus[:-1]]
    return np.array([
        0.9 * stimulus,
        0.6 * stimulus + 0.3 * previous,
        0.8 * stimulus**2,
        0.4 * stimulus + 0.4 * previous**2,
    ])


def save_figure(figure, folder, name):
    figure.savefig(folder / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_inputs(folder, first_image, *, ceiling, floor):
    time, signal = sine_input(120, dt=0.02, frequency=1.5, amplitude=0.35, offset=0.5)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, signal, label="Sine input", linewidth=2)
    for channel, trace in enumerate(synthetic_readouts(signal), 1):
        ax.plot(time, trace, label=f"Demo readout {channel}")
    ax.set(title="Synthetic sine-wave check", xlabel="Time", ylabel="Signal")
    ax.legend(ncols=2)
    save_figure(fig, folder, "sine-check")

    stimulus = encode_pixels(first_image, ceiling=ceiling, floor=floor)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5), layout="constrained")
    titles = ["Encoded input", *(f"Demo readout {channel}" for channel in range(1, 5))]
    for ax, trace, title in zip(axes, [stimulus, *synthetic_readouts(stimulus)], titles):
        ax.imshow(trace.reshape(28, 28), vmin=0, vmax=ceiling)
        ax.set_title(title)
        ax.set_axis_off()
    save_figure(fig, folder, "readout-inspection")


def plot_analysis(folder, pixels, features, labels, summaries, scores):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, matrix, title in zip(axes, (pixels, features), ("Input pixels", "Demo L2 features")):
        projected = pca_view(matrix)
        for digit in summaries:
            points = projected[labels == digit]
            ax.scatter(points[:, 0], points[:, 1], label=f"Digit {digit}")
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
    axes[1].legend()
    fig.suptitle("PCA views with synthetic readouts")
    save_figure(fig, folder, "pca")

    fig, (spread, profile, distance) = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    digits = [str(digit) for digit in summaries]
    spread.bar(digits, [item["spread"] for item in summaries.values()])
    spread.set(title="Within-class RMS spread", xlabel="Digit")
    for digit, item in summaries.items():
        profile.errorbar(range(1, 5), item["mean"], yerr=item["std"], label=str(digit), marker="o")
    profile.set(title="Mean L2 ± sample SD", xlabel="Demo readout", xticks=range(1, 5))
    profile.legend(title="Digit")
    fig.colorbar(distance.imshow(centroid_distances(summaries)), ax=distance)
    distance.set(title="Centroid distances", xticks=range(len(digits)), xticklabels=digits,
                 yticks=range(len(digits)), yticklabels=digits)
    fig.suptitle("Descriptive statistics with synthetic readouts")
    save_figure(fig, folder, "class-statistics")

    fig, ax = plt.subplots(layout="constrained")
    bars = ax.bar(["Input pixels", "Demo L2 features"], scores)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-1.1, 1.1)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(title="Silhouette comparison with synthetic readouts", ylabel="Silhouette score")
    save_figure(fig, folder, "silhouette")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=("pilot", "expanded"), default="expanded")
    parser.add_argument("--mnist-csv", type=Path, help="Optional local, headerless MNIST-format CSV")
    parser.add_argument("--output", type=Path, default=Path("demo-output"))
    parser.add_argument("--batch-size", type=int, help="Limit new images per invocation")
    args = parser.parse_args()

    if args.mnist_csv is None:
        pixels, labels = synthetic_images()
        source_id = "synthetic-patterns-v1"
    else:
        pixels, labels = load_mnist_csv(args.mnist_csv)
        with args.mnist_csv.open("rb") as stream:
            source_id = file_digest(stream, "sha256").hexdigest()

    if args.design == "pilot":
        indices = choose_images(labels, (1, 2, 3), 5)
    else:
        indices = choose_images(labels, (1, 2, 3, 4), 20, np.random.default_rng(42))
    selected_pixels, selected_labels = pixels[indices], labels[indices]

    output = args.output / args.design
    settings = {"ceiling": 1.0, "floor": 0.05}
    features = collect_features(
        selected_pixels, selected_labels, indices, synthetic_readouts, **settings,
        checkpoint=output / "demo-checkpoint.npz", source_id=f"demo-readouts-v1:{source_id}",
        batch_size=args.batch_size,
    )
    print(f"Demo progress: {len(features)}/{len(indices)} images")
    if len(features) < len(indices):
        return

    np.savetxt(
        output / "l2-features.csv",
        np.column_stack((indices, selected_labels, features)),
        delimiter=",",
        fmt=("%d", "%d", "%.8f", "%.8f", "%.8f", "%.8f"),
        header="input_row,label,l2_readout_1,l2_readout_2,l2_readout_3,l2_readout_4",
        comments="",
    )
    summaries = class_statistics(features, selected_labels)
    scores = silhouette_comparison(selected_pixels, features, selected_labels)
    print(f"Synthetic readout comparison: input {scores[0]:.4f}, L2 {scores[1]:.4f}")
    plot_inputs(output, selected_pixels[0], **settings)
    plot_analysis(output, selected_pixels, features, selected_labels, summaries, scores)
    print(f"Demo plots saved to {output}")


if __name__ == "__main__":
    main()
