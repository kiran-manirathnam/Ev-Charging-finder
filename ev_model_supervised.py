# ev_model_supervised.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------- Load the clustered dataset ----------------
DATASET = "bengaluru_ev_charging_clustered.csv"  # Make sure this file exists
df = pd.read_csv(DATASET)

# ---------------- Check dataset ----------------
print("Columns:", df.columns)
if 'cluster' not in df.columns:
    raise ValueError("Cluster column not found! Run the clustering script first.")

# ---------------- Features & Target ----------------
X = df[['latitude', 'longitude']]  # Input features
y = df['cluster']                  # Target: cluster number

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Feature scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train KNN Classifier ----------------
knn = KNeighborsClassifier(n_neighbors=5)  # k=5 nearest clusters
knn.fit(X_train_scaled, y_train)

# ---------------- Test accuracy ----------------
accuracy = knn.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")

# ---------------- Save model & scaler ----------------
joblib.dump(knn, "ev_knn_model.pkl")
joblib.dump(scaler, "ev_scaler.pkl")
print("Model and scaler saved as 'ev_knn_model.pkl' and 'ev_scaler.pkl'")

# ---------------- Example prediction ----------------
# Replace with your user's coordinates
user_coords = [[12.9716, 77.5946]]  # Bangalore center example
user_scaled = scaler.transform(user_coords)
predicted_cluster = knn.predict(user_scaled)[0]
print(f"Predicted cluster for {user_coords[0]}: {predicted_cluster}")
