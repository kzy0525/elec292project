import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Dummy Model (replace with real model)
def load_trained_model():
    model = LogisticRegression(max_iter=500)
    model.fit([[0]*30, [1]*30], [0,1])
    return model

# === Feature Extraction from Window ===
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

# === GUI ===
class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("ELEC 292 Windowed Walking / Jumping Classifier")
        master.geometry("700x600")

        self.model = load_trained_model()
        self.predictions = None

        tk.Label(master, text="Windowed CSV Classifier", font=("Arial", 20)).pack(pady=10)

        self.load_button = tk.Button(master, text="Load CSV", font=("Arial", 12), command=self.load_csv)
        self.load_button.pack(pady=5)

        self.save_button = tk.Button(master, text="Save Prediction CSV", font=("Arial", 12), command=self.save_csv, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(pady=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select Raw CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        df = pd.read_csv(file_path)

        # Rename columns if necessary
        if "Acceleration x (m/s^2)" in df.columns:
            df.rename(columns={
                "Acceleration x (m/s^2)": "x",
                "Acceleration y (m/s^2)": "y",
                "Acceleration z (m/s^2)": "z"
            }, inplace=True)

        if not all(col in df.columns for col in ['x', 'y', 'z']):
            messagebox.showerror("Error", "CSV must contain columns: x, y, z")
            return

        # === Segment into 5-second windows ===
        sample_rate = 50
        window_size = 5 * sample_rate

        segments = []
        for start in range(0, len(df) - window_size + 1, window_size):
            window = df.iloc[start:start+window_size]
            features = extract_features(window)
            segments.append(features)

        X = np.array(segments)

        # === Predict per window ===
        y_pred = self.model.predict(X)
        labels = ["walking" if pred == 0 else "jumping" for pred in y_pred]

        # === Prepare prediction table ===
        self.predictions = pd.DataFrame({
            "Window #": np.arange(1, len(labels) + 1),
            "Predicted Class": labels
        })

        # === Plot ===
        self.ax.clear()
        counts = self.predictions["Predicted Class"].value_counts()
        self.ax.bar(counts.index, counts.values, color=['royalblue', 'skyblue'])
        self.ax.set_title("Predicted Class Distribution")
        self.ax.set_ylabel("Window Count")
        self.ax.set_xlabel("Class")
        self.canvas.draw()

        self.save_button.config(state=tk.NORMAL)
        messagebox.showinfo("Done", "Prediction complete. You can now export.")

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