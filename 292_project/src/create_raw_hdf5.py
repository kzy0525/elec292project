import os
import pandas as pd
import numpy as np
import h5py

# Paths
RAW_DATA_DIR = "../data/raw"
OUTPUT_HDF5_PATH = "../data/accelerometer_all.h5"

# Mapping position code from filename to readable format
POSITION_MAP = {
    "HH": "hand",
    "PP": "pants",
    "JP": "jacket"
}

def parse_label(filename):
    return 0 if "walking" in filename.lower() else 1  # walking = 0, jumping = 1

def create_hdf5_from_csvs():
    with h5py.File(OUTPUT_HDF5_PATH, "w") as hdf:
        raw_group = hdf.create_group("raw")

        for participant in os.listdir(RAW_DATA_DIR):
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            if not os.path.isdir(participant_path):
                continue

            for filename in os.listdir(participant_path):
                if not filename.endswith(".csv"):
                    continue

                filepath = os.path.join(participant_path, filename)
                df = pd.read_csv(filepath)

                # Auto rename columns for consistency
                if "Acceleration x (m/s^2)" in df.columns:
                    df.rename(columns={
                        "Time (s)": "time",
                        "Acceleration x (m/s^2)": "x",
                        "Acceleration y (m/s^2)": "y",
                        "Acceleration z (m/s^2)": "z"
                    }, inplace=True)

                # Add label column
                label = parse_label(filename)
                df["label"] = label

                # Get position from filename (e.g., HH, JP, etc.)
                suffix = filename.split("_")[-1].replace(".csv", "")
                position = POSITION_MAP.get(suffix, "unknown")

                # Build HDF5 path and save
                group_path = f"raw/{participant}/{position}"
                raw_group.require_group(group_path)
                dataset_path = f"{group_path}/{filename.replace('.csv', '')}"
                hdf.create_dataset(dataset_path, data=df.to_numpy(), compression="gzip")
                print(f"Saved: {dataset_path} → shape {df.shape}")

    print(f"\nAll CSVs stored into: {OUTPUT_HDF5_PATH}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_hdf5_from_csvs()
