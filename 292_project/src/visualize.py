import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path to the HDF5 file (relative to src/)
hdf5_path = "data/accelerometer_data.h5"

# 🔹 Plot X, Y, Z acceleration vs. time
def plot_acceleration(data, title):
    plt.figure(figsize=(10, 5))
    time = np.arange(len(data))
    plt.plot(time, data[:, 0], label="X-axis")
    plt.plot(time, data[:, 1], label="Y-axis")
    plt.plot(time, data[:, 2], label="Z-axis")
    plt.xlabel("Sample Index (Time)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 🔹 Visualize all walking segments for Kevin (jacket, pants, hand)
def visualize_kevin_walking():
    position_map = {
        "HH": "hand",
        "JP": "jacket",
        "PP": "pants"
    }

    with h5py.File(hdf5_path, "r") as hdf:
        for code, pos in position_map.items():
            path = f"raw/kevin/{pos}/walking"
            if path in hdf:
                data = hdf[path][:]
                title = f"Kevin - {pos.capitalize()} - Walking"
                plot_acceleration(data, title)
            else:
                print(f"⚠️  Missing dataset: {path}")

# 🔹 Compare raw vs. preprocessed data (for a specific axis)
def compare_raw_vs_preprocessed(position="jacket", activity="walking", participant="kevin", axis=0):
    """
    Compare raw and preprocessed accelerometer signals for one axis (0 = x, 1 = y, 2 = z)
    """
    with h5py.File(hdf5_path, "r") as hdf:
        raw_path = f"raw/{participant}/{position}/{activity}"
        pre_path = f"preprocessed/{participant}/{position}/{activity}"

        if raw_path not in hdf or pre_path not in hdf:
            print(f"❌ Dataset not found: {raw_path} or {pre_path}")
            return

        raw = hdf[raw_path][:]
        clean = hdf[pre_path][:]

        time = np.arange(len(raw))
        axis_name = ['X', 'Y', 'Z'][axis]

        plt.figure(figsize=(12, 5))
        plt.plot(time, raw[:, axis], label=f"Raw {axis_name}", alpha=0.5)
        plt.plot(time, clean[:, axis], label=f"Smoothed {axis_name}", linewidth=2)
        plt.title(f"{axis_name}-Axis - Raw vs. Preprocessed\n{participant.title()} - {position.title()} - {activity.title()}")
        plt.xlabel("Sample Index")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 🔸 MAIN 🔸
if __name__ == "__main__":
    # Visualize all walking positions for Kevin
    visualize_kevin_walking()

    # Compare raw vs. preprocessed (change position/activity/axis if needed)
    compare_raw_vs_preprocessed(position="jacket", activity="walking", axis=0)
    #compare_raw_vs_preprocessed(position="pants", activity="walking", axis=1)
    # compare_raw_vs_preprocessed(position="hand", activity="walking", axis=2)
