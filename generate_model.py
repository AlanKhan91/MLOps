from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle, os

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("app", exist_ok=True)
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
