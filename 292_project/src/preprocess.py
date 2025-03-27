import h5py
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os

# Paths
input_path = "data/accelerometer_data.h5"
output_path = "data/accelerometer_data.h5"  # We'll update in-place

# Parameters
WINDOW_SIZE = 50  # for moving average smoothing


def preprocess_data(raw_data):
    # Fill missing values with linear interpolation
    df = pd.DataFrame(raw_data, columns=["x", "y", "z"])
    df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Apply moving average filter
    smoothed = uniform_filter1d(df.values, size=WINDOW_SIZE, axis=0)
    return smoothed


def process_all_kevin_data():
    with h5py.File(input_path, "r+") as hdf:
        for position in hdf["raw/kevin"]:
            for activity in hdf[f"raw/kevin/{position}"]:
                raw_dataset_path = f"raw/kevin/{position}/{activity}"
                raw_data = hdf[raw_dataset_path][:]

                # Preprocess the data
                cleaned_data = preprocess_data(raw_data)

                # Build output path
                group_path = f"preprocessed/kevin/{position}"
                dataset_path = f"{group_path}/{activity}"

                # Create group if it doesn't exist
                if group_path not in hdf:
                    hdf.create_group(group_path)

                # Delete old dataset if it exists (for reruns)
                if dataset_path in hdf:
                    del hdf[dataset_path]

                # Save cleaned data
                hdf.create_dataset(dataset_path, data=cleaned_data)
                print(f"✅ Preprocessed and saved: {dataset_path}")


if __name__ == "__main__":
    process_all_kevin_data()
