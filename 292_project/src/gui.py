import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import joblib
from sklearn.preprocessing import StandardScaler
def moving_average_filter(data):
    data = data.to_numpy()
    smoothed = np.copy(data)
    for i in range(1, 4):  # x, y, z

        smoothed[:, i] = pd.Series(data[:, i]).rolling(window=10, min_periods=1, center=True).mean()
    return smoothed

def normalize_polling_frequency(data):
    # Convert the time column to datetime if it isn't already
    print(data)
    df = pd.DataFrame(data, columns=['time', 'x', 'y', 'z'])

    df.index = pd.to_timedelta(df['time'], unit='s')
    interval_ms = 10  # milliseconds per sample
    if interval_ms.is_integer():
        interval_str = f"{int(interval_ms)}ms"
    else:
        # If the interval is not an integer number of milliseconds, create a Timedelta.
        interval_str = pd.Timedelta(milliseconds=interval_ms)

    # Resample the DataFrame using the calculated interval.
    # Here, we use the mean as the aggregation function, but you can change this if needed.
    downsampled = df.resample(interval_str).mean()

    # Optionally, add the time back as a column in seconds.
    downsampled['time'] = downsampled.index.total_seconds()

    # Reset the index if you prefer to work with a regular DataFrame.
    downsampled = downsampled.reset_index(drop=True)
    print(downsampled)
    return downsampled

# load the trained model from the train.py file to use for the GUI
def load_trained_model():
    if not os.path.exists("model.joblib"):
        raise FileNotFoundError("Trained model (model.joblib) not found! Please run train.py first.")
    return joblib.load("model.joblib")


#extracts features from the input csv file because the model requires 10 features from the training
#but the input csv does not have those
def extract_features(window):
    features = []
    for axis in ['x', 'y', 'z']:
        data = window[axis].values
        axis_features = [
            np.max(data), np.min(data), np.ptp(data), np.mean(data), np.median(data),
            np.var(data), np.std(data), skew(data),
            np.sqrt(np.mean(data ** 2)), np.mean(np.abs(data - np.mean(data)))
        ]
        features.extend(axis_features)
    return features

# GUI Class
class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("ELEC 292 Windowed Walking / Jumping Classifier")
        master.geometry("700x600")

        self.model = load_trained_model()
        self.predictions = None
        self.file_path = None
        self.scaler = StandardScaler()

        tk.Label(master, text="Elec 292 Project", font=("Arial", 20)).pack(pady=10)
        tk.Label(master, text="Jumping / Walking Classifier Model", font=("Arial", 10)).pack(pady=2)

        # button to select input file
        self.load_button = tk.Button(master, text="Load CSV", font=("Arial", 12), command=self.load_csv)
        self.load_button.pack(pady=5)

        # display the selected file name
        self.file_label_var = tk.StringVar()
        self.file_label_var.set("No file selected")
        self.file_label = tk.Label(master, textvariable=self.file_label_var, font=("Arial", 10), fg="gray")
        self.file_label.pack(pady=5)

        # button to save output CSV file
        self.save_button = tk.Button(master, text="Save Prediction CSV", font=("Arial", 12), command=self.save_csv, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # create the plot area but not visible until input csv is selected
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master)

    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        self.file_path = file_path
        self.file_label_var.set(f"Loaded: {os.path.basename(file_path)}")

        df = pd.read_csv(file_path)

        # auto rename columns, so it can be processed with the model
        if "Acceleration x (m/s^2)" in df.columns:
            df.rename(columns={
                "Acceleration x (m/s^2)": "x",
                "Acceleration y (m/s^2)": "y",
                "Acceleration z (m/s^2)": "z"
            }, inplace=True)

        if not all(col in df.columns for col in ['x', 'y', 'z']):
            messagebox.showerror("Error", "CSV must contain columns: x, y, z")
            return
        df = df.drop('Absolute acceleration (m/s^2)', axis=1)
        df = normalize_polling_frequency(df.to_numpy())
        df = moving_average_filter(df)
        df = pd.DataFrame(df, columns=['time', 'x', 'y', 'z'])
        # segment into 5 second windows
        sample_rate = 100
        window_size = 5 * sample_rate

        segments = []
        for start in range(0, len(df) - window_size + 1, window_size):
            window = df.iloc[start:start+window_size]
            features = extract_features(window)
            segments.append(features)
        print(segments)
        X = np.array(segments)
        X = self.scaler.fit_transform(X)
        # classify walking/jumping for eah 5 second window
        y_pred = self.model.predict(X.tolist())
        print('x is ', X)
        print(y_pred)
        labels = ["walking" if pred == 0 else "jumping" for pred in y_pred]

        # create the graph
        self.predictions = pd.DataFrame({
            "Window #": np.arange(1, len(labels) + 1),
            "Predicted Class": labels
        })

        # create scatter plot for either walking or jumping
        self.ax.clear()
        y_numeric = [1 if label == "jumping" else 0 for label in self.predictions["Predicted Class"]]

        # colour code it red for jumping and blue for walking
        colors = ['red' if y == 1 else 'blue' for y in y_numeric]
        self.ax.scatter(self.predictions["Window #"], y_numeric,
                        c=colors, marker='o', alpha=0.8)
        self.ax.set_title("Activity vs Time (Window Index)")
        self.ax.set_ylabel("Activity")
        self.ax.set_xlabel("Window #")
        self.ax.set_yticks([0, 1])
        self.ax.set_yticklabels(["Walking (Blue)", "Jumping (Red)"])
        self.ax.grid(True)
        self.fig.tight_layout(pad=3.0)

        # creates the graph visual only after the processing is complete
        if not hasattr(self, 'canvas_shown') or not self.canvas_shown:
            self.canvas.get_tk_widget().pack(pady=10)
            self.canvas_shown = True
        self.canvas.draw()
        self.save_button.config(state=tk.NORMAL)
        messagebox.showinfo("Done", "Classification complete. You can now export.")

    def save_csv(self):
        if self.predictions is None:
            messagebox.showwarning("No Predictions", "Please run predictions first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.predictions.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Saved to {save_path}")

root = tk.Tk()
app = PredictionApp(root)
root.mainloop()
