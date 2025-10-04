# ⚡ Bengaluru EV Charging Station Finder & ML Model

This project is an intelligent chatbot + machine learning model that helps users find **nearby EV charging stations** in Bengaluru.  
It combines geolocation search, Open Charge Map API integration, and clustering-based ML insights.

---

## 🚀 Features

- 🔍 Find nearest EV charging stations by locality
- 🌍 View stations on Google Maps directly
- 🔌 Fetch charger type and operator from Open Charge Map API
- 🧠 ML model (KNN + KMeans) to cluster and predict station areas
- 📊 Distance and location-based recommendations
- ⚙️ Built using **Streamlit**, **Scikit-learn**, and **Geopy**

---

## 🧩 Project Structure

```
📁 Bengaluru_EV_Project
│
├── bengaluru_ev_charging.csv          # Original datase
├── bengaluru_ev_charging_clustered.csv # Clustered dataset (output)
│
├── ev_model.py                        # K-Means clustering model
├── ev_model_supervised.py             # KNN supervised learning model
├── ev_chatbot_knn.py                  # Streamlit chatbot with model + API integration
│
├── ev_knn_model.joblib                # Trained ML model
├── ev_scaler.joblib                   # Scaler used for normalization
│
└── README.md                          # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/bengaluru-ev-finder.git
   cd bengaluru-ev-finder
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Open Charge Map API Key**
   - Open `ev_chatbot_knn.py`
   - Replace:
     ```python
     API_KEY = "YOUR_API_KEY_HERE"
     ```

4. **Run the chatbot**
   ```bash
   streamlit run ev_chatbot_knn.py
   ```

---

## 📦 Dependencies

```
streamlit
pandas
geopy
scikit-learn
requests
joblib
```

---

## 🧠 Model Overview

- **Clustering:** Uses K-Means to group EV stations based on latitude and longitude.
- **Supervised Learning:** KNN predicts the cluster of a new location for faster lookups.
- **Geopy:** Calculates real-world distances using haversine/geodesic distance.

---

## 🌐 API Integration

The chatbot fetches charger details from the **Open Charge Map API**:
- Charger type (AC/DC, kW)
- Operator information
- Real-time map link

---

## 📍 Example Usage

- Enter locality: `Indiranagar`
- Output:
  - Nearest stations (distance in km)
  - Charger type (from API)
  - Google Maps link for navigation

---

## 🧑‍💻 Author

Developed by **Kiran & Mayur**  

---

## 📜 License

This project is open-source under the **MIT License**.
