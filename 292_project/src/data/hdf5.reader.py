import h5py

# Unified HDF5 file path
hdf5_path = "../../data/accelerometer_all.h5"


def explore_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        print("📂 File structure:")
        hdf.visititems(lambda name, obj: print(f"  - {name}"))

        print("\n🧪 Available participant data in Raw data:")
        for participant in hdf["raw"]:
            for position in hdf[f"raw/{participant}"]:
                for activity in hdf[f"raw/{participant}/{position}"]:
                    path = f"raw/{participant}/{position}/{activity}"
                    data = hdf[path][:5]  # Preview first 5 rows
                    print(f"\n🔹 {path} → Shape: {hdf[path].shape}")
                    print(data)

if __name__ == "__main__":
    explore_hdf5_file(hdf5_path)
