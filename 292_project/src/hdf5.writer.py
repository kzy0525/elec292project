import os
import pandas as pd
import numpy as np
import h5py

# Path to the raw CSV folders by participant
CSV_ROOT_FOLDER = "../data/raw/"
HDF5_OUTPUT_PATH = "data/accelerometer_data.h5"


def clean_column_names(df):
    """Standardizes column names to: time, x, y, z, abs (if available)."""
    return df.rename(columns={
        "Time (s)": "time",
        "Acceleration x (m/s^2)": "x",
        "Acceleration y (m/s^2)": "y",
        "Acceleration z (m/s^2)": "z",
        "Absolute acceleration (m/s^2)": "abs"
    })


def parse_metadata(filename):
    """Infers activity and position from filename."""
    fname = filename.lower()

    # Determine activity
    if "walking" in fname:
        activity = "walking"
    elif "jumping" in fname:
        activity = "jumping"
    else:
        raise ValueError(f"Unknown activity in filename: {filename}")

    # Determine position
    if "hand" in fname or "hh" in fname:
        position = "hand"
    elif "pants" in fname or "pp" in fname:
        position = "pants"
    elif "jacket" in fname or "jp" in fname:
        position = "jacket"
    else:
        raise ValueError(f"Unknown position in filename: {filename}")

    return activity, position


def write_to_hdf5():
    os.makedirs(os.path.dirname(HDF5_OUTPUT_PATH), exist_ok=True)

    with h5py.File(HDF5_OUTPUT_PATH, "w") as hdf:
        for participant in os.listdir(CSV_ROOT_FOLDER):
            participant_path = os.path.join(CSV_ROOT_FOLDER, participant)
            if not os.path.isdir(participant_path):
                continue  # Skip files in /raw/ folder root

            for filename in os.listdir(participant_path):
                if not filename.endswith(".csv"):
                    continue

                file_path = os.path.join(participant_path, filename)

                try:
                    # Load and clean data
                    df = pd.read_csv(file_path)
                    df = clean_column_names(df)

                    # Ensure x, y, z columns exist
                    if not all(col in df.columns for col in ["x", "y", "z"]):
                        print(f"⚠️ Skipped {filename}: missing x/y/z columns")
                        continue

                    # Add label: 0 = walking, 1 = jumping
                    activity, position = parse_metadata(filename)
                    label = 1 if activity == "jumping" else 0
                    df["label"] = label

                    # Select columns to save
                    data = df[["time", "x", "y", "z", "label"]].to_numpy(dtype=np.float32)


                    # HDF5 dataset path
                    dataset_path = f"raw/{participant}/{position}/{activity}"
                    hdf.create_dataset(dataset_path, data=data, compression="gzip")
                    print(f"✅ Saved: {dataset_path} ({filename})")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")


if __name__ == "__main__":
    write_to_hdf5()
