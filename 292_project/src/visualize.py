import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path to the HDF5 file
hdf5_path = "data/accelerometer_data.h5"

# 🔹 Plot X, Y, Z acceleration vs. time
def plot_acceleration(data, title):
    plt.figure(figsize=(10, 5))
    time = data[:, 0]
    plt.plot(time, data[:, 1], label="X-axis")
    plt.plot(time, data[:, 2], label="Y-axis")
    plt.plot(time, data[:, 3], label="Z-axis")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_3d_acceleration_scatter(walking_data, jumping_data):
    """
    walking_data and jumping_data are numpy arrays with at least columns:
    [time, x, y, z]
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot walking data in blue
    ax.scatter(walking_data[:, 1], walking_data[:, 2], walking_data[:, 3],
               c='dodgerblue', alpha=0.4, label='Walking')

    # Plot jumping data in red
    ax.scatter(jumping_data[:, 1], jumping_data[:, 2], jumping_data[:, 3],
               c='crimson', alpha=0.4, label='Jumping')

    ax.set_xlabel("X Acceleration (m/s²)")
    ax.set_ylabel("Y Acceleration (m/s²)")
    ax.set_zlabel("Z Acceleration (m/s²)")
    ax.set_title("3D Scatter Plot of Acceleration")
    ax.legend()
    plt.tight_layout()
    plt.show()

# 🔹 Visualize walking data for any participant and all positions
def visualize_acceleration_for_participant(participant):
    with h5py.File(hdf5_path, "r") as hdf:
        base_path = f"raw/{participant}"
        if base_path not in hdf:
            print(f"❌ No data found for participant: {participant}")
            return

        for position in hdf[base_path]:
            for activity in ["walking", "jumping"]:
                activity_path = f"{base_path}/{position}/{activity}"
                if activity_path in hdf:
                    data = hdf[activity_path][:]
                    title = f"{participant.title()} - {position.title()} - {activity.title()}"
                    plot_acceleration(data, title)
                else:
                    print(f"⚠️  No {activity} data for {participant} at {position}")


def visualize_3d_scatter_for(participant, position):
    with h5py.File(hdf5_path, "r") as hdf:
        walking_path = f"raw/{participant}/{position}/walking"
        jumping_path = f"raw/{participant}/{position}/jumping"

        if walking_path in hdf and jumping_path in hdf:
            walking = hdf[walking_path][:]
            jumping = hdf[jumping_path][:]
            plot_3d_acceleration_scatter(walking, jumping)
        else:
            print(f"❌ One or both datasets not found for {participant} at {position}")

def visualize_histograms_for(participant, position):
    with h5py.File(hdf5_path, "r") as hdf:
        walking_path = f"raw/{participant}/{position}/walking"
        jumping_path = f"raw/{participant}/{position}/jumping"
        if walking_path in hdf and jumping_path in hdf:
            walking = hdf[walking_path][:]
            jumping = hdf[jumping_path][:]
            plot_histograms_per_axis(walking, jumping)
        else:
            print(f"❌ Cannot plot histograms: Missing data for {participant} at {position}")

def plot_histograms_per_axis(walking_data, jumping_data, bins=60):
    axis_names = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    for i in range(3):
        axs[0, i].hist(walking_data[:, i + 1], bins=bins, color='dodgerblue', alpha=0.7)
        axs[0, i].set_title(f"Walking - Acceleration {axis_names[i]}")
        axs[0, i].set_xlabel("Acceleration (m/s²)")
        axs[0, i].set_ylabel("Frequency")

        axs[1, i].hist(jumping_data[:, i + 1], bins=bins, color='crimson', alpha=0.7)
        axs[1, i].set_title(f"Jumping - Acceleration {axis_names[i]}")
        axs[1, i].set_xlabel("Acceleration (m/s²)")
        axs[1, i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# 🔹 Compare raw vs. preprocessed data (for a specific axis)
def compare_raw_vs_preprocessed(participant="kevin", position="jacket", activity="walking", axis=0):
    raw_path = f"raw/{participant}/{position}/{activity}"
    pre_path = f"preprocessed/{participant}/{position}/{activity}"

    with h5py.File("data/accelerometer_data.h5", "r") as raw_hdf, \
         h5py.File("data/accelerometer_preprocessed.h5", "r") as pre_hdf:

        if raw_path not in raw_hdf or pre_path not in pre_hdf:
            print(f"❌ Path missing in one of the files:\n{raw_path}\n{pre_path}")
            return

        raw = raw_hdf[raw_path][:]
        clean = pre_hdf[pre_path][:]

        time = raw[:, 0]  # Assumes time values are the same
        axis_name = ['X', 'Y', 'Z'][axis]

        plt.figure(figsize=(12, 5))
        plt.plot(time, raw[:, axis + 1], label=f"Raw {axis_name}", alpha=0.5)
        plt.plot(time, clean[:, axis + 1], label=f"Smoothed {axis_name}", linewidth=2)
        plt.title(f"{axis_name}-Axis: Raw vs. Preprocessed\n{participant.title()} - {position.title()} - {activity.title()}")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 🔸 MAIN 🔸
if __name__ == "__main__":

    compare_raw_vs_preprocessed(participant="kevin", position="hand", activity="walking", axis=0)
    # View walking data for any participant (change the name below)
    visualize_acceleration_for_participant("kevin")
    #visualize_walking_for_participant("evan")
    #visualize_walking_for_participant("simon")

    # Visualize 3D scatter for a participant-position pair
    visualize_3d_scatter_for("kevin", "hand")

    # Compare raw vs. preprocessed (make sure preprocessing has been done)
    # compare_raw_vs_preprocessed(participant="kevin", position="jacket", activity="walking", axis=0)

    visualize_histograms_for("kevin", "hand")