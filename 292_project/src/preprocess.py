import numpy as np
import pandas as pd
import h5py


# path to the CSV folders for the raw data input and preprocessed data output
raw_hdf5_path = "data/accelerometer_data.h5"
pre_hdf5_path = "data/accelerometer_preprocessed.h5"

# window size used for the average smoothing, able to adjust based on goal (between 5 and 15ish)
WINDOW_SIZE = 10

# minimum polling rate - different devices had different polling rates, this makes sures they're all downscaled to one rate
POLLING_FREQ = 100
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

def normalize_polling_frequency(data):
    # Extract the time column (assumed to be in seconds) and ensure it's sorted
    times = data[:, 0]

    # Determine the start and end times for the data
    t_start, t_end = times[0], times[-1]

    sampling_interval = 1.0 / 100
    new_times = np.arange(t_start, t_end, sampling_interval)
    downsampled_data = np.empty((len(new_times), data.shape[1]))

    # The new time column is the first column of the downsampled data.
    downsampled_data[:, 0] = new_times

    # For each additional column, interpolate values at the new time stamps.
    for col in range(data.shape[1]):
        if col == 0:
            continue
        downsampled_data[:, col] = np.interp(new_times, times, data[:, col])

    return downsampled_data

# saves the smoothed and filled in data into the "preprocessed" hdf5 file
def preprocess_and_save():
    with h5py.File(raw_hdf5_path, "r") as raw_hdf, h5py.File(pre_hdf5_path, "w") as pre_hdf:
        pre_group = pre_hdf.create_group("preprocessed")

        for participant in raw_hdf["raw"]:
            for position in raw_hdf["raw"][participant]:
                for activity in raw_hdf["raw"][participant][position]:
                    path = f"raw/{participant}/{position}/{activity}"
                    data = raw_hdf[path][:]
                    # preprocess: fill NaNs and apply moving average
                    data = fill_missing(data)
                    data = normalize_polling_frequency(data)
                    data = moving_average_filter(data)

                    # Save to new HDF5
                    group_path = f"preprocessed/{participant}/{position}"
                    pre_hdf.require_group(group_path)
                    pre_hdf.create_dataset(f"{group_path}/{activity}", data=data, compression="gzip")
                    print(f"Saved to new file: {group_path}/{activity}")

if __name__ == "__main__":
    preprocess_and_save()
