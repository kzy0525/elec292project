import h5py
import numpy as np
import random
import os

input_path = "data/accelerometer_data.h5"
output_path = "data/accelerometer_data.h5"

WINDOW_SIZE = 500  # 5 seconds at 100 Hz
TRAIN_SPLIT = 0.9

def segment_and_store():
    X = []
    y = []

    with h5py.File(input_path, "r+") as hdf:
        pre_path = "preprocessed/kevin"

        # Loop through each position and activity
        for position in hdf[pre_path]:
            for activity in hdf[f"{pre_path}/{position}"]:
                label = 0 if activity == "walking" else 1
                data = hdf[f"{pre_path}/{position}/{activity}"][:]

                # Split data into non-overlapping 5s windows
                num_windows = len(data) // WINDOW_SIZE
                for i in range(num_windows):
                    window = data[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
                    X.append(window)
                    y.append(label)

        print(f"📦 Total Segments: {len(X)}")

        # Shuffle dataset
        combined = list(zip(X, y))
        random.shuffle(combined)
        X[:], y[:] = zip(*combined)

        # Split into train/test
        split_idx = int(len(X) * TRAIN_SPLIT)
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]

        # Delete old segmented data if rerunning
        if "segmented_data" in hdf:
            del hdf["segmented_data"]
        segmented = hdf.create_group("segmented_data")
        train_grp = segmented.create_group("train")
        test_grp = segmented.create_group("test")

        # Save
        train_grp.create_dataset("X", data=np.array(train_X))
        train_grp.create_dataset("y", data=np.array(train_y))
        test_grp.create_dataset("X", data=np.array(test_X))
        test_grp.create_dataset("y", data=np.array(test_y))

        print(f"✅ Train: {len(train_X)} | Test: {len(test_X)}")

if __name__ == "__main__":
    segment_and_store()
