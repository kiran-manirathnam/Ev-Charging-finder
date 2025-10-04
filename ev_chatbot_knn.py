# ev_chatbot_knn.py
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import joblib
import requests

# ---------------- Load Dataset ----------------
DATASET = "bengaluru_ev_charging_clustered.csv"
df = pd.read_csv(DATASET)

# ---------------- Load KNN Model & Scaler ----------------
knn_model = joblib.load("ev_knn_model.pkl")
scaler = joblib.load("ev_scaler.pkl")

# ---------------- Geocoding Helper ----------------
def get_coordinates(place_name):
    """Convert locality name to latitude and longitude using geopy + OpenStreetMap."""
    geolocator = Nominatim(user_agent="ev_locator")
    location = geolocator.geocode(place_name + ", Bengaluru, India", timeout=10)
    if location:
        return (location.latitude, location.longitude)
    return None

# ---------------- Open Charge Map API ----------------
OCM_API_KEY = "311f41d3-a9eb-4c38-afa7-41e0a48c394d"

def get_charger_type(lat, lon):
    """Fetch charger type from Open Charge Map API."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "distance": 0.5,
        "distanceunit": "KM",
        "maxresults": 1,
        "key": OCM_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data and "Connections" in data[0]:
            types = [conn.get("ConnectionType", {}).get("Title", "Unknown") for conn in data[0]["Connections"]]
            return ", ".join(types)
    except:
        pass
    return "Unknown"

# ---------------- Nearest Stations using KNN ----------------
def find_nearest_stations(user_location, top_n=5):
    """Predict cluster using KNN, then return nearest stations in that cluster."""
    lat, lon = user_location
    X_scaled = scaler.transform([[lat, lon]])
    cluster_label = knn_model.predict(X_scaled)[0]

    cluster_stations = df[df['cluster'] == cluster_label].copy()
    distances = []
    for _, row in cluster_stations.iterrows():
        station_coords = (row['latitude'], row['longitude'])
        distance = geodesic(user_location, station_coords).km
        distances.append(distance)
    cluster_stations['distance'] = distances
    return cluster_stations.sort_values(by='distance').head(top_n)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="EV Charging Station Finder - Bengaluru", page_icon="⚡", layout="centered")

st.title("⚡ Bengaluru EV Charging Station Finder")
st.markdown("Find the nearest EV charging stations in Bengaluru by entering your locality name.")

# User input
user_input = st.text_input("Enter your locality in Bengaluru:", "")

if user_input:
    coords = get_coordinates(user_input)
    if not coords:
        st.error("❌ Could not find that locality. Please try again with a different name.")
    else:
        st.success(f"📍 Location found: {user_input} ({coords[0]:.4f}, {coords[1]:.4f})")
        nearest = find_nearest_stations(coords, top_n=5)

        st.subheader("🔌 Nearest Charging Stations:")
        for _, row in nearest.iterrows():
            maps_url = f"https://www.google.com/maps/search/?api=1&query={row['latitude']},{row['longitude']}"
            # Fetch charger type dynamically
            charger_type = get_charger_type(row['latitude'], row['longitude'])
            open_time = row['open_time_info'] if 'open_time_info' in row and pd.notna(row.get('open_time_info')) else ''

            st.markdown(
                f"**{row['name']}**  \n"
                f"📍 {row['address']}  \n"
                f"🕒 {open_time}  \n"
                f"⚡ Charger Type: {charger_type}  \n"
                f"🛣 Distance: {row['distance']:.2f} km  \n"
                f"[Open in Google Maps]({maps_url})"
            )
