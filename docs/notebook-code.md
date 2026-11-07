# SSP Notebook code excerpts

These are source excerpts matched against the local notebooks and the supplied screenshots. The numbers below are zero-based cell positions in the saved files. Jupyter execution numbers shown in screenshots depend on run order. The extracts preserve the analysis statements. A few sections begin after setup or omit the adjacent simulator or data-loading block. Each excerpt expects variables from earlier cells.

For a complete runnable walkthrough, use [the demo](../analysis/demo.py) and [analysis functions](../analysis/workflow.py). Those files adapt the workflow to user-supplied or synthetic inputs. They include a separate illustrative readout provider.

## Screenshot coverage

| Images in upload order | What they show | Where to look |
| --- | --- | --- |
| 1–3, 11, 14, 16–18 | Imports, missing dependency, CSV path, preprocessing and notebook state | [Setup and debugging](#setup-and-debugging) |
| 4–10, 12–13, 19–20 | Encoded inputs, simulator call boundary, trace plotting, and reshaping a readout for inspection | [Signal preparation and inspection](#signal-preparation-and-inspection) |
| 15 | Pilot sample selection | [Original pilot cell](#pilot-sample-selection) |
| 21–23 | Expanded counts, per-image simulation, checkpoint and the first saved batch | [Batch processing and checkpoints](#batch-processing-and-checkpoints) |
| 24–29 | Class statistics, profiles, spread, centroid distances, and the corrected silhouette chart | [Results code and chart correction](#results-code-and-chart-correction) |
| 30–33 | The separate PyTorch teaching example with a sine wave, reference input, and four readouts | [Signal preparation and inspection](#signal-preparation-and-inspection) |

## Setup and debugging

The screenshots show a missing dependency, a CSV path that did not resolve, and cells run before their imports or setup. The runnable version has explicit dependencies and takes the CSV as a command-line argument. It reads the first column as the label, scales pixel values to 0–1 and uses all rows of a headerless file. The old loader inferred a header and dropped the first row, which accounts for the 59,999 rows shown in the screenshots. The original result comparisons refer to the archived sample selections.

## Signal preparation and inspection

The teaching notebooks provided the simulator interface. The runnable version implements pixel-to-signal encoding, an illustrative sine input and trace visualisations. A readout provider receives one stimulus and returns four time-aligned channels. The group solver and its configuration stay outside this repository. The reference input in the screenshots was a zero-voltage signal. One old plot labelled it “Drain current”.

## Batch processing and checkpoints

The expanded notebook resets the network state for each image and saves one feature row at a time. The adapted [checkpoint code](../analysis/workflow.py) verifies sample order, labels and settings before resuming. It writes the updated checkpoint atomically. The original screenshot with a syntax error shows an unfinished draft cell, and a later screenshot shows a successful saved batch.

## Results code and chart correction

The original class summaries below use sample standard deviations and Euclidean distances between class centroids. PCA is a two-dimensional view. The silhouette calculation uses the full standardised spaces. The early silhouette plot clipped the negative bar. The runnable [plotting code](../analysis/demo.py) includes a zero line and displays the full score range. All demo plots are generated locally.

## Notebook extracts

### Single-image L2 calculation

`nanowire_reservoir_L2.ipynb`, cell position 19.

```python
input_img = signals[0, :784].reshape(28, 28)

for i in range(4):
    output_img = readout_voltages[i, :784].reshape(28, 28)
    l2 = np.linalg.norm(input_img - output_img)
    print(f"Electrode {i}: L2 norm = {l2:.4f}")
```

### L2 summary across all four readouts

`nanowire_reservoir_L2.ipynb`, cell position 20.

```python
l2_values = []

for i in range(4):
    output_img = readout_voltages[i, :784].reshape(28, 28)
    l2 = np.linalg.norm(input_img - output_img)
    l2_values.append(l2)

plt.bar(range(4), l2_values)
plt.xlabel("Readout electrode")
plt.ylabel("L2 norm")
plt.title("Difference between MNIST input and NWN readout")
plt.xticks(range(4))
plt.show()
```

### Pilot sample selection

`nanowire_reservoir_L2.ipynb`, cell position 21.

```python
classes = [1, 2, 3]
samples_per_class = 5

selected_indices = np.concatenate([
    np.where(label_train == c)[0][:samples_per_class]
    for c in classes
])

selected_data = train_data[selected_indices]
selected_labels = label_train[selected_indices]

print("Selected indices:", selected_indices)
print("Selected labels:", selected_labels)
print("Data shape:", selected_data.shape)
```

### Pilot image grid

`nanowire_reservoir_L2.ipynb`, cell position 22.

```python
fig, axes = plt.subplots(3, 5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(selected_data[i].reshape(28, 28))
    ax.set_title(f"Digit {selected_labels[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### PCA of the readout features

`nanowire_reservoir_L2.ipynb`, cell position 25.

```python
# Centre the 4-D L2 features
X = all_l2_features
X_centered = X - X.mean(axis=0)

# PCA: reduce 4 dimensions to 2 dimensions
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_pca = X_centered @ Vt[:2].T

# Plot digits 1, 2 and 3
for digit in [1, 2, 3]:
    mask = selected_labels == digit
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Digit {digit}",
        s=80
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA clustering of NWN L2 features")
plt.legend()
plt.grid()
plt.show()
```

### PCA of the original pixel representation

`nanowire_reservoir_L2.ipynb`, cell position 26.

```python
# PCA on raw MNIST images for comparison

X_raw = selected_data
X_raw_centered = X_raw - X_raw.mean(axis=0)

U_raw, S_raw, Vt_raw = np.linalg.svd(
    X_raw_centered,
    full_matrices=False
)

X_raw_pca = X_raw_centered @ Vt_raw[:2].T

for digit in [1, 2, 3]:
    mask = selected_labels == digit
    plt.scatter(
        X_raw_pca[mask, 0],
        X_raw_pca[mask, 1],
        label=f"Digit {digit}",
        s=80
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA clustering of raw MNIST images")
plt.legend()
plt.grid()
plt.show()
```

### Silhouette scores in separately standardised spaces

`nanowire_reservoir_L2.ipynb`, cell position 27.

```python
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# NWN L2 features
X_nwn_scaled = StandardScaler().fit_transform(all_l2_features)

# Raw MNIST features
X_raw_scaled = StandardScaler().fit_transform(selected_data)

score_nwn = silhouette_score(X_nwn_scaled, selected_labels)
score_raw = silhouette_score(X_raw_scaled, selected_labels)

print("Silhouette score - Raw MNIST:", round(score_raw, 4))
print("Silhouette score - NWN L2 features:", round(score_nwn, 4))
```

### Expanded sample selection

`nanowire_reservoir_final_analysis.ipynb`, cell position 15.

```python
final_classes = [1, 2, 3, 4]
final_samples_per_class = 20

rng = np.random.default_rng(42)

final_indices = np.concatenate([
    rng.choice(
        np.where(label_train == c)[0],
        size=final_samples_per_class,
        replace=False
    )
    for c in final_classes
])

final_data = train_data[final_indices]
final_labels = label_train[final_indices]

print("Data shape:", final_data.shape)
print("Total images:", len(final_labels))

for c in final_classes:
    print(f"Digit {c}: {np.sum(final_labels == c)} images")
```

### One feature row per image

`nanowire_reservoir_final_analysis.ipynb`, cell position 16.

```python
l2_features = []

for i in range(4):

    l2 = np.linalg.norm(
        input_signal - readout_single[i, :784]
    )

    l2_features.append(l2)

final_l2_list.append(l2_features)
```

### Class summaries and RMS spread

`nanowire_reservoir_final_analysis.ipynb`, cell position 17.

```python
assert X.shape == (80, 4)
assert y.shape == (80,)
assert np.isfinite(X).all()

print("\n===== Class Statistics =====")

for digit in [1, 2, 3, 4]:

    Xi = X[y == digit]

    mean = Xi.mean(axis=0)
    std = Xi.std(axis=0, ddof=1)
    variance = Xi.var(axis=0, ddof=1)

    spread = np.sqrt(
        np.mean(np.sum((Xi - mean)**2, axis=1))
    )

    print(f"\nDigit {digit}")
    print("Samples:", len(Xi))
    print("Mean L2:", np.round(mean, 4))
    print("Standard deviation:", np.round(std, 4))
    print("Variance:", np.round(variance, 4))
    print("Within-class spread:", round(spread, 4))
```

### Mean profiles and variability plots

`nanowire_reservoir_final_analysis.ipynb`, cell position 18.

```python
import matplotlib.pyplot as plt
import numpy as np

digits = [1, 2, 3, 4]

means = np.array([
    X[y == d].mean(axis=0) for d in digits
])

stds = np.array([
    X[y == d].std(axis=0, ddof=1) for d in digits
])

spreads = np.array([
    np.sqrt(np.mean(np.sum(
        (X[y == d] - X[y == d].mean(axis=0))**2,
        axis=1
    )))
    for d in digits
])

# Figure 1: Within-class spread
plt.figure(figsize=(8, 5))
plt.bar([str(d) for d in digits], spreads)

for i, value in enumerate(spreads):
    plt.text(i, value + 0.015, f"{value:.4f}",
             ha="center")

plt.xlabel("MNIST Digit")
plt.ylabel("Within-class RMS Spread")
plt.title("NWN Feature Variability by Digit")
plt.ylim(0, max(spreads) * 1.18)
plt.tight_layout()
plt.savefig("final_within_class_spread.png", dpi=300)
plt.show()


# Figure 2: Mean L2 with standard deviation
plt.figure(figsize=(9, 5))

for i, digit in enumerate(digits):
    plt.errorbar(
        range(1, 5),
        means[i],
        yerr=stds[i],
        marker="o",
        capsize=4,
        label=f"Digit {digit}"
    )

plt.xlabel("Readout Electrode")
plt.ylabel("Mean L2 Distance")
plt.title("NWN Mean L2 Response with Standard Deviation")
plt.xticks(range(1, 5))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("final_electrode_mean_sd.png", dpi=300)
plt.show()
```

### Euclidean distances between class centres

`nanowire_reservoir_final_analysis.ipynb`, cell position 19.

```python
import numpy as np
import matplotlib.pyplot as plt


# X = features
# y = labels

digits = [1, 2, 3, 4]

# 1. （centroid）
centroids = []
for d in digits:
    centroid = X[y == d].mean(axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# 2.
dist_matrix = np.zeros((len(digits), len(digits)))

for i in range(len(digits)):
    for j in range(len(digits)):
        dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

print("Centroid distance matrix:")
print(np.round(dist_matrix, 4))

# 3.
plt.figure(figsize=(6, 5))
plt.imshow(dist_matrix, cmap="viridis")
plt.colorbar(label="Euclidean distance")

plt.xticks(range(len(digits)), [f"Digit {d}" for d in digits])
plt.yticks(range(len(digits)), [f"Digit {d}" for d in digits])
plt.title("Centroid distance heatmap of NWN features")


for i in range(len(digits)):
    for j in range(len(digits)):
        plt.text(j, i, f"{dist_matrix[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.tight_layout()
plt.savefig("final_centroid_distance_heatmap.png", dpi=300)
plt.show()
```

### Standardisation before the expanded comparison

`nanowire_reservoir_final_analysis.ipynb`, cell position 20.

```python
X_nwn_scaled = StandardScaler().fit_transform(X_nwn)
X_raw_scaled = StandardScaler().fit_transform(X_raw)

# ===== 3. 计算 silhouette score =====
score_raw = silhouette_score(X_raw_scaled, y)
score_nwn = silhouette_score(X_nwn_scaled, y)

print("Silhouette score - Raw MNIST:", round(score_raw, 4))
print("Silhouette score - NWN features:", round(score_nwn, 4))
```

### Silhouette chart with both signs visible

`nanowire_reservoir_final_analysis.ipynb`, cell position 21. The fixed axis limits were removed so the plot scales to the inputs.

```python
scores = [score_raw, score_nwn]
labels = ["Raw MNIST", "NWN L2 features"]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, scores)

plt.axhline(0, color="black", linewidth=1)

plt.ylabel("Silhouette Score")
plt.title("Class Separability: Raw MNIST vs NWN L2 Features")

for bar, value in zip(bars, scores):
    x = bar.get_x() + bar.get_width() / 2

    if value >= 0:
        plt.text(x, value + 0.004, f"{value:.4f}", ha="center")
    else:
        plt.text(x, value - 0.007, f"{value:.4f}", ha="center")

plt.tight_layout()
plt.savefig("final_silhouette_comparison.png", dpi=300)
plt.show()
```

## Provided teaching notebook excerpts

These extracts show the input preparation and visual inspection used in the provided examples. The research group supplied the teaching notebooks. The excerpts refer to caller-provided variables and leave out device configuration, model initialisation and outputs.

### Separate labels and normalise MNIST pixels

`nanowire_reservoir_example.ipynb`, cell position 5.

```python
# labels are in first column
label_train = mnist_train[mnist_train.columns[0]].to_numpy()
train_data = mnist_train.loc[:,:].to_numpy()
train_data = train_data[:,1:]/255

label_test = mnist_test[mnist_test.columns[0]].to_numpy()
test_data = mnist_test.loc[:,:].to_numpy()
test_data = test_data[:,1:]/255
```

### Scale and floor the voltage sequence

`nanowire_reservoir_example.ipynb`, cell position 12.

```python
train_V = Vmax*train_data[:n_train,:].flatten()
for i in range(len(train_V)):
    train_V[i] = max(train_V[i],Vmin)

input_signal = train_V[:len(time_vector)]
signals = np.vstack((input_signal, drain))
```

### Construct a sine-wave input

`nanowire_reservoir_example.ipynb`, cell position 13.

```python
time_vector = np.arange(0, 784*dt, dt)

input_signal = offset + A*np.sin(2*np.pi*frequency*time_vector)

drain = np.zeros(len(time_vector))

signals = np.vstack((input_signal, drain))
```

### Inspect the four readouts as images

`nanowire_reservoir_example.ipynb`, cell position 21.

```python
index = 1
plt.subplot(1,5,1)

plt.imshow(signals[0,(784*index):(784*index)+784].reshape((28,28)))
plt.title("raw MNIST \n digit")
plt.subplot(1,5,2)
plt.imshow(readout_voltages[0,(784*index):(784*index)+784].reshape((28,28)))
plt.title("readout \n electrode 0")
# plt.show()
plt.subplot(1,5,3)
plt.imshow(readout_voltages[1,(784*index):(784*index)+784].reshape((28,28)))
plt.title("readout \n electrode 1")
# plt.show()
plt.subplot(1,5,4)
plt.imshow(readout_voltages[2,(784*index):(784*index)+784].reshape((28,28)))
plt.title("readout \n electrode 2")
# plt.show()
plt.subplot(1,5,5)
plt.imshow(readout_voltages[3,(784*index):(784*index)+784].reshape((28,28)))
plt.title("readout \n electrode 3")
plt.show()
```

### Stack a PyTorch signal and zero-voltage reference

`nanowire_network_example.ipynb`, cell position 6.

```python
input_signal = sine_wave + noise

drain = torch.zeros(len(time_vector))

signals = torch.vstack((input_signal, drain))
```
