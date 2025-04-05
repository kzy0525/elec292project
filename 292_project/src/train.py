import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os
import joblib

# paths
DEFAULT_FEATURES_CSV = "data/features_normalized.csv"
OUTPUT_PREDICTIONS_CSV = "data/predictions.csv"

# accept csv from command
if len(sys.argv) > 1:
    FEATURES_CSV = sys.argv[1]
else:
    FEATURES_CSV = DEFAULT_FEATURES_CSV

# load the features
print(f"Loading features from {FEATURES_CSV}...")
df = pd.read_csv(FEATURES_CSV)
X = df.drop("label", axis=1).values if "label" in df.columns else df.values
y = df["label"].values if "label" in df.columns else None

# split the model into 90:10 ration of training and testing, then classify
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"✅ Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # training the logistic regression model
    model = LogisticRegression(max_iter=500)

    model.fit(X_train, y_train)

    # check the accuracy of the model on the test files
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    # prints classification report using sklearn
    print(classification_report(y_test, y_pred, target_names=["Walking", "Jumping"]))
    print("Confusion Matrix:")
    # prints confusion matrix using sklearn
    print(confusion_matrix(y_test, y_pred))

    # finds the values needed for the learning curve plot
    train_sizes, train_scores, test_scores = learning_curve(
        LogisticRegression(max_iter=500), X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    # plots the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Accuracy", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Validation Accuracy", marker='x')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save model predictions to CSV
    df_out = pd.DataFrame(X_test, columns=df.drop("label", axis=1).columns)
    df_out["true_label"] = y_test
    df_out["predicted_label"] = y_pred
    df_out["predicted_class"] = ["walking" if label == 0 else "jumping" for label in y_pred]
    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    #print(f"Predictions saved to {OUTPUT_PREDICTIONS_CSV}")

    print("")
# edge cases
else:
    print("No labels found in input CSV. Skipping training, predicting only.")
    # Load model (could be replaced with pre-trained model loader)
    model = LogisticRegression(max_iter=500)
    model.fit(X, np.zeros(X.shape[0]))  # Dummy training
    y_pred = model.predict(X)
    df_out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df_out["predicted_label"] = y_pred
    df_out["predicted_class"] = ["walking" if label == 0 else "jumping" for label in y_pred]
    df_out.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_PREDICTIONS_CSV}")

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# prints all the data needed for the report like test sizes and accuracies
print(f"Train set size: {len(X_train)} samples ({len(X_train) / len(X) * 100:.2f}%)")
print(f"Test set size: {len(X_test)} samples ({len(X_test) / len(X) * 100:.2f}%)")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# saves model which can be called in the GUI
print("")
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")

