# segment_windows.py
import h5py
import numpy as np
import os

# Paths
PRE_HDF = "data/accelerometer_preprocessed.h5"
WIN_HDF = "data/windowed_segments.h5"

WINDOW_SIZE = 5      # seconds
SAMPLE_RATE = 50     # samples per second
STEP = WINDOW_SIZE * SAMPLE_RATE

def segment_data(data, step):
    return [data[i:i+step] for i in range(0, len(data) - step + 1, step)
            if data[i:i+step].shape[0] == step and not np.any(np.isnan(data[i:i+step]))]

def segment_and_save():
    train_segments = {'walking': [], 'jumping': []}
    test_segments = {'walking': [], 'jumping': []}

    with h5py.File(PRE_HDF, "r") as pre_hdf:
        for participant in pre_hdf["preprocessed"]:
            for position in pre_hdf["preprocessed"][participant]:
                for activity in pre_hdf["preprocessed"][participant][position]:
                    data = pre_hdf[f"preprocessed/{participant}/{position}/{activity}"][:]
                    label = "walking" if int(data[0, 4]) == 0 else "jumping"
                    windows = segment_data(data[:, 1:4], STEP)  # x, y, z only
                    train_segments[label].extend(windows)

    # Shuffle and split
    all_windows = {}
    for label in ["walking", "jumping"]:
        np.random.shuffle(train_segments[label])
        split = int(len(train_segments[label]) * 0.9)
        all_windows[f"train/{label}"] = train_segments[label][:split]
        all_windows[f"test/{label}"] = train_segments[label][split:]

    # Write to HDF5
    with h5py.File(WIN_HDF, "w") as win_hdf:
        for key, segments in all_windows.items():
            group = win_hdf.require_group(key)
            for i, window in enumerate(segments):
                group.create_dataset(f"segment_{i}", data=window, compression="gzip")
                print(f"✅ Saved {key}/segment_{i} [{window.shape}]")

    print(f"\n🎉 All segmented data saved to {WIN_HDF}")

if __name__ == "__main__":
    segment_and_save()

