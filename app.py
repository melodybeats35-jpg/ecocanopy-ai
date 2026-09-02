import streamlit as st
import xgboost as xgb
import pandas as pd

st.set_page_config(page_title="EcoCanopy AI", page_icon="🌳", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("ecocanopy_model.json")
    return model

@st.cache_data
def load_city_data():
    return pd.read_csv("real_world_cities_dataset.csv")

model = load_model()
city_df = load_city_data()

st.markdown("# 🌳 EcoCanopy AI")
st.markdown("### Urban Heat Island Mitigation & Spatial Green Infrastructure Planner")
st.markdown(
    "<span style='color:gray; font-size:0.9em;'>"
    "Data sources: NASA POWER (temperature) · Google Earth Engine / Sentinel-2 (NDVI) · "
    "World Population Review & national census bureaus (population density)"
    "</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🗺️ Zone Selection Panel")
mode = st.sidebar.radio("Selection Mode:", ["🌐 Choose from Real Global Cities", "✍️ Custom Zone Input"])

if mode == "🌐 Choose from Real Global Cities":
    city_list = sorted(city_df["City"].tolist())
    selected_city = st.sidebar.selectbox("Select City:", city_list)

    row = city_df[city_df["City"] == selected_city].iloc[0]
    zone_data = {
        "Concrete_Density": float(row["Concrete_Density"]),
        "NDVI_Vegetation_Index": float(row["NDVI_Vegetation_Index"]),
        "Population_Density": float(row["Population_Density"])
    }
    zone_label = selected_city

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Real Zone Feature Values")
    st.sidebar.markdown(f"**Concrete Density:** {zone_data['Concrete_Density']:.1f}%")
    st.sidebar.markdown(f"**NDVI Vegetation Index:** {zone_data['NDVI_Vegetation_Index']:.3f}")
    st.sidebar.markdown(f"**Population Density:** {int(zone_data['Population_Density'])} people/km²")
    st.sidebar.markdown(f"**Recorded Ambient Temp (NASA):** {row['Ambient_Temperature']:.2f} °C")
    st.sidebar.markdown(f"**Coordinates:** {row['Latitude']:.3f}, {row['Longitude']:.3f}")

else:
    st.sidebar.markdown("### ✍️ Apna Zone Data Enter Karein")
    custom_area_name = st.sidebar.text_input("Area / Location Name", value="My Custom Zone")
    concrete_input = st.sidebar.number_input(
        "Concrete Density (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.5
    )
    ndvi_input = st.sidebar.number_input(
        "NDVI Vegetation Index (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.20, step=0.01
    )
    population_input = st.sidebar.number_input(
        "Population Density (people/km²)", min_value=0.0, max_value=30000.0, value=6000.0, step=100.0
    )
    zone_data = {
        "Concrete_Density": concrete_input,
        "NDVI_Vegetation_Index": ndvi_input,
        "Population_Density": population_input
    }
    zone_label = custom_area_name if custom_area_name.strip() else "Custom Zone"

st.sidebar.markdown("---")
st.sidebar.markdown("### About EcoCanopy AI")
st.sidebar.markdown(
    "This planner uses an XGBoost regression engine trained on real satellite and "
    "climate reanalysis data across 24 global cities to forecast localized ambient "
    "temperature anomalies and recommend green infrastructure interventions."
)

# ---------------- MAIN PANEL ----------------
input_df = pd.DataFrame([{
    "Concrete_Density": zone_data["Concrete_Density"],
    "NDVI_Vegetation_Index": zone_data["NDVI_Vegetation_Index"],
    "Population_Density": zone_data["Population_Density"]
}])

predicted_temp = float(model.predict(input_df)[0])

st.markdown(f"## 📍 Selected Location: {zone_label}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Concrete Density", value=f"{zone_data['Concrete_Density']:.1f}%")
with col2:
    st.metric(label="NDVI Vegetation Index", value=f"{zone_data['NDVI_Vegetation_Index']:.3f}")
with col3:
    st.metric(label="Population Density", value=f"{int(zone_data['Population_Density'])} /km²")

st.markdown("---")
st.markdown("## 🌡️ Predictive Micro-Climate Output")
st.metric(label=f"Predicted Ambient Temperature — {zone_label}", value=f"{predicted_temp:.2f} °C")

if predicted_temp > 38.0:
    st.error(
        "🔴 **CRITICAL PRIORITY 1**\n\n"
        "Deploy urban tree canopies and albedo interventions immediately. "
        "This zone exhibits severe heat island intensity requiring high-density "
        "green infrastructure and reflective surface retrofitting."
    )
elif predicted_temp > 35.0:
    st.warning(
        "🟠 **WARNING PRIORITY 2**\n\n"
        "Increase vegetative wall matrices by 15%. "
        "This zone shows elevated thermal stress and would benefit from "
        "targeted vertical greening and shade infrastructure."
    )
else:
    st.success(
        "🟢 **STABLE CLIMATE SEGMENT**\n\n"
        "This zone is within acceptable ambient temperature thresholds. "
        "Maintain current vegetation coverage and monitor seasonal drift."
    )

# ---------------- GLOBAL DATA TABLE ----------------
st.markdown("---")
st.markdown("## 🏙️ Spatial Hierarchy — Real Global City Dataset")

display_df = city_df[[
    "City", "Latitude", "Longitude", "Concrete_Density",
    "NDVI_Vegetation_Index", "Population_Density", "Ambient_Temperature"
]].copy()
display_df = display_df.rename(columns={
    "Concrete_Density": "Concrete_%",
    "NDVI_Vegetation_Index": "NDVI",
    "Population_Density": "Pop_Density",
    "Ambient_Temperature": "Recorded_Temp_C"
})
display_df = display_df.sort_values("Recorded_Temp_C", ascending=False).reset_index(drop=True)

st.dataframe(display_df, use_container_width=True, height=420)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "EcoCanopy AI — GeoAI Spatial Intelligence Platform | Predictive Engine: XGBoost Regressor | "
    "Trained on real satellite/climate data — 24 global cities | "
    "Sources: NASA POWER, Sentinel-2 (GEE), World Population Review"
    "</div>",
    unsafe_allow_html=True
)
