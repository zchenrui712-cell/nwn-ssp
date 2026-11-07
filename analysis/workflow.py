"""SSP analysis stages adapted to run independently of the group simulator."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_mnist_csv(path: Path):
    """Read a headerless CSV with a digit label followed by 784 pixels."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.shape[1] != 785 or not np.isfinite(rows).all():
        raise ValueError("Expected a finite label and 784 pixels per row")
    labels, pixels = rows[:, 0], rows[:, 1:]
    if not np.all((labels == np.floor(labels)) & (labels >= 0) & (labels <= 9)):
        raise ValueError("Labels must be integers from 0 to 9")
    if np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("Pixels must be between 0 and 255")
    return pixels / 255, labels.astype(int)


def choose_images(labels, digits, per_digit, rng=None):
    """Take first matches for the pilot or sample without replacement."""
    groups = []
    for digit in digits:
        candidates = np.flatnonzero(labels == digit)
        if candidates.size < per_digit:
            raise ValueError(f"Digit {digit} has fewer than {per_digit} images")
        selection = candidates[:per_digit] if rng is None else rng.choice(
            candidates, size=per_digit, replace=False
        )
        groups.append(selection)
    return np.concatenate(groups)


def encode_pixels(image, *, ceiling, floor):
    """Flatten an image and apply a lower bound to the scaled signal."""
    pixels = np.asarray(image, dtype=float).reshape(-1)
    if pixels.size != 784 or not np.isfinite(pixels).all():
        raise ValueError("Expected 784 finite pixel intensities")
    if np.any((pixels < 0) | (pixels > 1)) or not 0 <= floor <= ceiling:
        raise ValueError("Expected normalised pixels and 0 <= floor <= ceiling")
    return np.maximum(ceiling * pixels, floor)


def sine_input(steps, *, dt, frequency, amplitude, offset):
    time = np.arange(steps) * dt
    signal = offset + amplitude * np.sin(2 * np.pi * frequency * time)
    return time, signal


def readout_features(stimulus, readouts):
    """Compute four L2 distances from four aligned readout traces."""
    stimulus = np.asarray(stimulus, dtype=float)
    readouts = np.asarray(readouts, dtype=float)
    if stimulus.ndim != 1 or not stimulus.size or readouts.shape != (4, stimulus.size):
        raise ValueError("Expected one input trace and four equal-length readouts")
    if not np.isfinite(stimulus).all() or not np.isfinite(readouts).all():
        raise ValueError("Traces must be finite")
    return np.linalg.norm(stimulus - readouts, axis=1)


def collect_features(images, labels, indices, readout_provider, *, ceiling, floor,
                     checkpoint=None, source_id=None, batch_size=None):
    """Calculate features, optionally resuming a verified local checkpoint.

    The provider takes one stimulus and returns a fresh four-channel readout.
    Supply a stable source ID that identifies the input dataset and provider settings.
    """
    if checkpoint is not None and not source_id:
        raise ValueError("A source ID is required for checkpointing")
    if batch_size is not None and batch_size < 1:
        raise ValueError("Batch size must be positive")
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    if labels.shape != (len(images),) or indices.shape != labels.shape:
        raise ValueError("Images, labels and indices must describe the same rows")
    settings = np.array([ceiling, floor])
    features = []
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["indices"], indices)
                or not np.array_equal(saved["labels"], labels)
                or not np.array_equal(saved["settings"], settings)
                or saved["source_id"].item() != source_id
            ):
                raise ValueError("Checkpoint does not match this run")
            partial = saved["features"]
            if partial.ndim != 2 or partial.shape[1] != 4 or len(partial) > len(images):
                raise ValueError("Invalid checkpoint feature shape")
            if not np.isfinite(partial).all():
                raise ValueError("Checkpoint contains non-finite features")
            features = partial.tolist()

    start = len(features)
    stop = len(images) if batch_size is None else min(len(images), start + batch_size)
    for image in images[start:stop]:
        stimulus = encode_pixels(image, ceiling=ceiling, floor=floor)
        features.append(readout_features(stimulus, readout_provider(stimulus)))
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=checkpoint.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez_compressed(
                        stream, features=np.asarray(features), indices=indices,
                        labels=labels, settings=settings, source_id=source_id,
                    )
                    stream.close()
                    temporary.replace(checkpoint)
                finally:
                    stream.close()
                    temporary.unlink(missing_ok=True)
    return np.asarray(features, dtype=float).reshape(-1, 4)


def pca_view(matrix):
    """Centre features and project onto the first two right singular vectors."""
    centred = matrix - matrix.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return centred @ components[:2].T


def class_statistics(features, labels):
    summaries = {}
    for digit in np.unique(labels):
        samples = features[labels == digit]
        if len(samples) < 2:
            raise ValueError("Sample variance requires at least two images per class")
        mean = samples.mean(axis=0)
        summaries[int(digit)] = {
            "count": len(samples),
            "mean": mean,
            "std": samples.std(axis=0, ddof=1),
            "variance": samples.var(axis=0, ddof=1),
            "spread": np.sqrt(np.mean(np.sum((samples - mean) ** 2, axis=1))),
        }
    return summaries


def centroid_distances(summaries):
    means = np.array([item["mean"] for item in summaries.values()])
    return np.linalg.norm(means[:, None] - means[None, :], axis=2)


def silhouette_comparison(pixels, features, labels):
    """Score each full feature space after separately standardising columns."""
    return tuple(
        silhouette_score(StandardScaler().fit_transform(matrix), labels)
        for matrix in (pixels, features)
    )
