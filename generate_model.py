# generate_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train a RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save to app/iris_model.pkl
os.makedirs("app", exist_ok=True)
with open("app/iris_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Generated app/iris_model.pkl locally")
