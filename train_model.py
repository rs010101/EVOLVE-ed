import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("student_data.csv")

# Encode categorical variables
df["Engagement Level"] = df["Engagement Level"].map({"Low": 0, "Medium": 1, "High": 2})
df["Difficulty Level"] = df["Difficulty Level"].map({"Easy": 0, "Medium": 1, "Hard": 2, "Advanced": 3})

# Define features and labels
X = df[["Quiz Score", "Engagement Level"]]
y = df["Difficulty Level"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "adaptive_model.pkl")

print("Model trained and saved as adaptive_model.pkl")
