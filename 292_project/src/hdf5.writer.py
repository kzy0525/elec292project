import pandas as pd
import h5py
import os

# Base folder where the raw CSV files are located
csv_folder = "../data/raw/kevin/"


# Format: (participant_name, phone_position, activity_label, filename)
files = [
    ("kevin", "jacket", "walking", "kevin_walking_JP.csv"),
    ("kevin", "hand", "walking", "kevin_walking_HH.csv"),
    ("kevin", "pants", "walking", "kevin_walking_PP.csv"),
    #("participant1", "hand", "jumping", "p1_hand_jumping.csv"),
    # Repeat for participant2 and participant3...
]

# Renames long column headers to simpler names
def clean_column_names(df):
    return df.rename(columns={
        "Time (s)": "time",
        "Acceleration x (m/s^2)": "x",
        "Acceleration y (m/s^2)": "y",
        "Acceleration z (m/s^2)": "z",
        "Absolute acceleration (m/s^2)": "abs"
    })

# Writes all files into one HDF5 file
def write_to_hdf5(file_list, hdf5_path="data/accelerometer_data.h5"):
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)  # Ensure 'data/' exists

    with h5py.File(hdf5_path, "w") as hdf:
        for participant, position, activity, filename in file_list:
            # Build full file path
            full_path = os.path.join(csv_folder, filename)

            # Load and clean CSV
            df = pd.read_csv(full_path)
            df = clean_column_names(df)

            # Extract only x, y, z columns
            data = df[['x', 'y', 'z']].to_numpy()

            # Build path inside the HDF5 structure
            group_path = f"raw/{participant}/{position}/{activity}"

            # Save the dataset
            hdf.create_dataset(group_path, data=data)
            print(f"✅ Saved: {filename} → {group_path}")

# Run the script directly
if __name__ == "__main__":
    write_to_hdf5(files)
