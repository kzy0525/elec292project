import h5py
import numpy as np

# paths
PRE_HDF = "../data/accelerometer_all.h5"
WIN_HDF = "../data/accelerometer_all.h5"

WINDOW_SIZE = 5      # seconds
SAMPLE_RATE = 100     # samples per second
STEP = WINDOW_SIZE * SAMPLE_RATE

def segment_data(data, step):
    return [data[i:i+step] for i in range(0, len(data) - step + 1, step)
            if data[i:i+step].shape[0] == step and not np.any(np.isnan(data[i:i+step]))]

def segment_and_save():
    train_segments = {'walking': [], 'jumping': []}

    # reads the data from the proprocessed hdf5 file
    with h5py.File(PRE_HDF, "r") as pre_hdf:
        for participant in pre_hdf["Pre-processed data"]:
            for position in pre_hdf[f"Pre-processed data/{participant}"]:
                for activity in pre_hdf[f"Pre-processed data/{participant}/{position}"]:
                    data = pre_hdf[f"Pre-processed data/{participant}/{position}/{activity}"][:]
                    # creates a 5th label in the hdf5 for walking or jumping classification
                    label = "walking" if int(data[0, 4]) == 0 else "jumping"
                    # segments the continuous signal into smaller sliding windows
                    windows = segment_data(data[:, 1:4], STEP)  # x, y, z only
                    train_segments[label].extend(windows)

    # segments the test files and training files with the 90:10 split
    all_windows = {}
    for label in ["walking", "jumping"]:
        np.random.shuffle(train_segments[label])
        split = int(len(train_segments[label]) * 0.9)
        all_windows[f"train/{label}"] = train_segments[label][:split]
        all_windows[f"test/{label}"] = train_segments[label][split:]

    # function to write all the new segments into the hdf5 file
    with h5py.File(WIN_HDF, "a") as win_hdf:  # "a" = append to existing HDF5
        seg_group = win_hdf.require_group("Segmented data")
        for key, segments in all_windows.items():
            group = seg_group.require_group(key)
            for i, window in enumerate(segments):
                group.create_dataset(f"segment_{i}", data=window, compression="gzip")
                print(f"Saved {key}/segment_{i} [{window.shape}]")
    print(f"\nAll segmented data saved to {WIN_HDF}")

if __name__ == "__main__":
    segment_and_save()

# checking correct directory
with h5py.File("../data/accelerometer_all.h5", "r") as hdf:
    print("Top-level groups:")
    for key in hdf.keys():
        print(f" - {key}")
