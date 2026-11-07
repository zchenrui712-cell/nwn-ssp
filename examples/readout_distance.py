from math import dist

# Synthetic traces for a worked example.
stimulus = (0.0, 1.0, 0.0)
readouts = ((0.0, 0.75, 0.0), (0.25, 0.75, 0.25))

features = tuple(round(dist(stimulus, readout), 3) for readout in readouts)
print(features)
