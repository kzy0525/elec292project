import numpy as np
import pandas as pd
import h5py

# path to the CSV folders for the raw data input and preprocessed data output
hdf5_path = "../data/accelerometer_all.h5"


# window size used for the average smoothing, able to adjust based on goal (between 5 and 15ish)
WINDOW_SIZE = 10

# function that does the smoothing
def moving_average_filter(data, window=WINDOW_SIZE):
    smoothed = np.copy(data)
    for i in range(1, 4):  # x, y, z
        smoothed[:, i] = pd.Series(data[:, i]).rolling(window=window, min_periods=1, center=True).mean()
    return smoothed

# function in preprocessing to fill in gaps in data
def fill_missing(data):
    df = pd.DataFrame(data, columns=["time", "x", "y", "z", "label"])
    df.interpolate(method='linear',inplace = True)
    return df.to_numpy(dtype=np.float32)

# saves the smoothed and filled in data into the "preprocessed" hdf5 file
def preprocess_and_save():
    with h5py.File(hdf5_path, "a") as hdf:

        # Check if raw data exists
        if "raw" not in hdf:
            print("'Raw data' group not found in HDF5. Please run the raw data import script first.")
            return

        pre_group = hdf.require_group("Pre-processed data")

        for participant in hdf["raw"]:
            for position in hdf[f"raw/{participant}"]:
                for activity in hdf[f"raw/{participant}/{position}"]:
                    path = f"raw/{participant}/{position}/{activity}"
                    try:
                        data = hdf[path][:]
                        data = fill_missing(data)
                        data = moving_average_filter(data)

                        group_path = f"Pre-processed data/{participant}/{position}"
                        dataset_path = f"{group_path}/{activity}"

                        hdf.require_group(group_path)
                        if dataset_path in hdf:
                            del hdf[dataset_path]
                        hdf.create_dataset(dataset_path, data=data, compression="gzip")

                        print(f"Saved: {dataset_path}")
                    except Exception as e:
                        print(f"Failed to process {path}: {e}")

if __name__ == "__main__":
    preprocess_and_save()

with h5py.File("../data/accelerometer_all.h5", "r") as hdf:
    print("Top-level groups:")
    for key in hdf.keys():
        print(f" - {key}")
