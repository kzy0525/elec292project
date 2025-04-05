import h5py
import matplotlib.pyplot as plt

# path to the HDF5 file
hdf5_path = "data/accelerometer_data.h5"

# creates plots for all the jumping and walking data (hands, jacket pockets, pants pockets) for the specified person
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

def visualize_acceleration_for_participant(participant):
    with h5py.File(hdf5_path, "r") as hdf:
        base_path = f"raw/{participant}"
        if base_path not in hdf:
            print(f"No data found for participant: {participant}")
            return

        for position in hdf[base_path]:
            for activity in ["walking", "jumping"]:
                activity_path = f"{base_path}/{position}/{activity}"
                if activity_path in hdf:
                    data = hdf[activity_path][:]
                    title = f"{participant.title()} - {position.title()} - {activity.title()}"
                    plot_acceleration(data, title)
                else:
                    print(f"No {activity} data for {participant} at {position}")


# plots a 3d scatter plot for the walking and jumping data for visualization
def plot_3d_acceleration_scatter_separate(walking_data, jumping_data, participant, position):
    # --- Walking Plot ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(walking_data[:, 1], walking_data[:, 2], walking_data[:, 3],
                c='dodgerblue', alpha=0.5, label='Walking')
    ax1.set_xlabel("X Acceleration (m/s²)")
    ax1.set_ylabel("Y Acceleration (m/s²)")
    ax1.set_zlabel("Z Acceleration (m/s²)")
    ax1.set_title(f"3D Scatter Plot - Walking\nParticipant: {participant}, Position: {position}")
    ax1.legend()
    plt.tight_layout()

    # --- Jumping Plot ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(jumping_data[:, 1], jumping_data[:, 2], jumping_data[:, 3],
                c='crimson', alpha=0.5, label='Jumping')
    ax2.set_xlabel("X Acceleration (m/s²)")
    ax2.set_ylabel("Y Acceleration (m/s²)")
    ax2.set_zlabel("Z Acceleration (m/s²)")
    ax2.set_title(f"3D Scatter Plot - Jumping\nParticipant: {participant}, Position: {position}")
    ax2.legend()
    plt.tight_layout()

    plt.show()

def visualize_3d_scatter_for(participant, position):
    with h5py.File(hdf5_path, "r") as hdf:
        walking_path = f"raw/{participant}/{position}/walking"
        jumping_path = f"raw/{participant}/{position}/jumping"

        if walking_path in hdf and jumping_path in hdf:
            walking = hdf[walking_path][:]
            jumping = hdf[jumping_path][:]
            plot_3d_acceleration_scatter_separate(walking, jumping, participant, position)
        else:
            print(f"One or both datasets not found for {participant} at {position}")


# creates histograms for all the data based on frequency vs acceleration values
def visualize_histograms_for(participant, position):
    with h5py.File(hdf5_path, "r") as hdf:
        walking_path = f"raw/{participant}/{position}/walking"
        jumping_path = f"raw/{participant}/{position}/jumping"
        if walking_path in hdf and jumping_path in hdf:
            walking = hdf[walking_path][:]
            jumping = hdf[jumping_path][:]
            plot_histograms_per_axis(walking, jumping)
        else:
            print(f"Cannot plot histograms: Missing data for {participant} at {position}")

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
            print(f"Path missing in one of the files:\n{raw_path}\n{pre_path}")
            return

        raw = raw_hdf[raw_path][:]
        clean = pre_hdf[pre_path][:]
        time = raw[:, 0]
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

# main file to call on the visualization functions
if __name__ == "__main__":
    #comparing the smoothed preprocessed data vs raw data (x = 0, y = 1, z = 2)
    compare_raw_vs_preprocessed(participant="kevin", position="hand", activity="walking", axis=0)
    compare_raw_vs_preprocessed(participant="kevin", position="pants", activity="jumping", axis=1)
    compare_raw_vs_preprocessed(participant="evan", position="jacket", activity="walking", axis=2)

    #graphs for all the acceleration vs time graphs
    visualize_acceleration_for_participant("kevin")
    visualize_acceleration_for_participant("evan")
    visualize_acceleration_for_participant("simon")

    #graphs for 3d scatter plots
    visualize_3d_scatter_for("kevin", "hand")
    visualize_3d_scatter_for("evan", "jacket")

    #graphs for the frequency histograms
    visualize_histograms_for("kevin", "hand")
    visualize_histograms_for("evan", "jacket")