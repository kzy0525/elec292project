import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import os

# input output files
INPUT_HDF = "data/accelerometer_preprocessed.h5"
RAW_FEATURES_CSV = "data/features.csv"
NORMALIZED_FEATURES_CSV = "data/features_normalized.csv"

# calculating step for 5 second windows
WINDOW_SIZE = 5  # seconds
SAMPLE_RATE = 50  # samples/sec
STEP = WINDOW_SIZE * SAMPLE_RATE

# takes continuous time-series data and splits it into non-overlapping segments
def segment_data(data, step):
    return [data[i:i+step] for i in range(0, len(data) - step + 1, step)
            if data[i:i+step].shape[0] == step and not np.any(np.isnan(data[i:i+step]))]

# extracts the chosen 10 features for the training
def extract_features_from_window(window):
    features = []
    for axis in range(1, 4):  # x, y, z
        data = window[:, axis]
        if np.all(data == data[0]):
            # If all values are the same, skew is undefined
            skew_val = 0.0
        else:
            skew_val = skew(data)

        # uses numpy functions for the values
        features.extend([
            np.max(data), np.min(data), np.ptp(data), np.mean(data), np.median(data),
            np.var(data), np.std(data), skew_val,
            np.sqrt(np.mean(data ** 2)), np.mean(np.abs(data - np.mean(data)))
        ])

    return features

# extract all the features and create a new hdf5 file for each person, activity, and position
# with the features and values
def extract_features_for_all():
    rows = []
    with h5py.File(INPUT_HDF, "r") as hdf:
        for participant in hdf["preprocessed"]:
            for position in hdf["preprocessed"][participant]:
                for activity in hdf["preprocessed"][participant][position]:
                    path = f"preprocessed/{participant}/{position}/{activity}"
                    data = hdf[path][:]
                    label = int(data[0, 4])
                    windows = segment_data(data, STEP)
                    for window in windows:
                        features = extract_features_from_window(window)
                        features.append(label)
                        rows.append(features)

    col_names = [f"{stat}_{axis}" for axis in ['x', 'y', 'z']
                 for stat in ['max', 'min', 'range', 'mean', 'median', 'var', 'std', 'skew', 'rms', 'mad']]
    col_names.append("label")

    df = pd.DataFrame(rows, columns=col_names)
    os.makedirs(os.path.dirname(RAW_FEATURES_CSV), exist_ok=True)
    df.to_csv(RAW_FEATURES_CSV, index=False)
    # print statement to confirm completion of function
    print(f"Extracted features saved to {RAW_FEATURES_CSV}")

def normalize_features():
    df = pd.read_csv(RAW_FEATURES_CSV)
    features = df.drop("label", axis=1)
    labels = df["label"]
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(normalized, columns=features.columns)
    df_scaled["label"] = labels
    df_scaled.to_csv(NORMALIZED_FEATURES_CSV, index=False)
    # print statement to confirm completion of function
    print(f"Normalized features saved to {NORMALIZED_FEATURES_CSV}")

# main function to call the extraction functions created above
if __name__ == "__main__":
    extract_features_for_all()
    normalize_features()