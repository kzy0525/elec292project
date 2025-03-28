import h5py
import numpy as np
import pandas as pd
import os

# Input/output HDF5 paths
raw_hdf5_path = "data/accelerometer_data.h5"
pre_hdf5_path = "data/accelerometer_preprocessed.h5"

WINDOW_SIZE = 5


def moving_average_filter(data, window=WINDOW_SIZE):
    smoothed = np.copy(data)
    for i in range(1, 4):  # x, y, z
        smoothed[:, i] = pd.Series(data[:, i]).rolling(window=window, min_periods=1, center=True).mean()
    return smoothed


def fill_missing(data):
    df = pd.DataFrame(data, columns=["time", "x", "y", "z", "label"])
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df.to_numpy(dtype=np.float32)


def preprocess_and_save():
    with h5py.File(raw_hdf5_path, "r") as raw_hdf, h5py.File(pre_hdf5_path, "w") as pre_hdf:
        pre_group = pre_hdf.create_group("preprocessed")

        for participant in raw_hdf["raw"]:
            for position in raw_hdf["raw"][participant]:
                for activity in raw_hdf["raw"][participant][position]:
                    path = f"raw/{participant}/{position}/{activity}"
                    data = raw_hdf[path][:]

                    # Preprocess: fill NaNs and apply moving average
                    data = fill_missing(data)
                    data = moving_average_filter(data)

                    # Save to new HDF5
                    group_path = f"preprocessed/{participant}/{position}"
                    pre_hdf.require_group(group_path)
                    pre_hdf.create_dataset(f"{group_path}/{activity}", data=data, compression="gzip")
                    print(f"✅ Saved to new file: {group_path}/{activity}")


if __name__ == "__main__":
    preprocess_and_save()
