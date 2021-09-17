from pandas.core.algorithms import isin
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
	page_title = "Crop yield prediction",
	page_icon = "corn"
	)

st.title('Task 4 - Crop Yield Prediction')
with st.beta_expander("Problem Statement"):
    st.write("""To predict the yield of a specific crops by 
        using data such as rainfall, temperature, soil composition, 
        PH-values etc. """)
with st.beta_expander("Approach"):
    st.write("""To be filled """)
with st.beta_expander("Models used"):
    st.write("""To be filled """)

@st.cache
# To start with, just loading and returning
# We can show filtered details little later
def load_province_data():
    province_data = pd.read_csv("task-4-crop-yield-prediction/province_data.csv", delimiter=";")
    columns = province_data.columns
    print(columns)
    #filtered_province_data = province_data[['Geo Point', 'Provincie name', 'Gemeente name']]
    return province_data

@st.cache
def load_weather_stations_data():
    weather_stations_data = pd.read_csv("task-4-crop-yield-prediction/weather_knmi_stations_2018.csv", delimiter=";")
    return weather_stations_data

@st.cache
def load_weather_data():
    weather_data = pd.read_csv("task-4-crop-yield-prediction/weather_KNMI_20181231.csv", delimiter=";", nrows=500)
    return weather_data

@st.cache
def load_yield_data():
    yield_data = pd.read_csv("task-4-crop-yield-prediction/yield_data_cbs.csv", delimiter=";")
    yield_data = yield_data.iloc[:500]
    return yield_data

@st.cache
def load_weather_yield_data():
    weather_yield_data = pd.read_excel("task-4-crop-yield-prediction/merged_weather_yield_(3).xlsx")
    return weather_yield_data

def eda_year_yield_scatter(weather_yield_data, chosen_region, chosen_crop):
    # Line chart for crop-yield

    actual_years = weather_yield_data.year.unique()

    # Default is Potato
    internal_crop_name = ["PotatoesTotal_9"]

    # Likewise, we need to do mapping for all crops in dropdown
    if chosen_crop == "Arable Vegetables":
        internal_crop_name = "VegetablesArable_13"
    elif chosen_crop == "Cereals":
        internal_crop_name = "Cereals_14"
    elif chosen_crop == "Industrial Crops":
        internal_crop_name = "IndustrialCrops_16"
    elif chosen_crop == "Pulses":
        internal_crop_name = "Pulses_17"
    elif chosen_crop == "Sugar Beets":
        internal_crop_name = ["SugarBeets_18"]
    elif chosen_crop == "Other Arable Crops":
        internal_crop_name = "OtherArableCrops_19"
    elif chosen_crop == "Fruits":
        internal_crop_name = "FruitInTheOpen_38"
    elif chosen_crop == "Mushrooms":
        internal_crop_name = ["MushroomsTotal_68"]

    condition = ((weather_yield_data['NAME'] == chosen_region) & \
        (weather_yield_data['year'].isin(actual_years)) & \
        (weather_yield_data['FarmTypes'] == 'A009481'))
    filtered_yield = weather_yield_data.loc[condition, internal_crop_name]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        actual_years,
        filtered_yield,
    )
    ax.set_xlabel("Year")
    ax.set_xticklabels(actual_years, rotation=45, ha="right")
    ax.set_ylabel("Crop yield (in millions per hectare)")
    
    st.write(fig)

def predict_yield():
    return "Work in progress!"

# Explore each dataset that is used in this task
st.sidebar.subheader("Datasets Used")
if st.sidebar.button("Explore Provinces Dataset"):
    province_data = load_province_data()
    st.write("Quick look of Provinces data set, more from a geographical location perspective")
    st.write("Data source: ")
    st.write(province_data)
    st.write("Meta Data: ")

if st.sidebar.button("Explore Weather Stations Dataset"):
    weather_stations_data = load_weather_stations_data()
    st.write("Weather stations in Netherlands (2018 data)")
    st.write("Data source: https://www.kaggle.com/sinaasappel/historical-weather-in-the-netherlands-19012018 ")
    st.write(weather_stations_data)
    st.write("Meta Data:")

if st.sidebar.button("Explore Weather Dataset"):
    weather_data = load_weather_data()
    st.write("Historical weather in the Netherlands collected from 1901 to 2018 (Displaying first 500 records)")
    st.write("Data source: https://www.kaggle.com/sinaasappel/historical-weather-in-the-netherlands-19012018 ")
    st.write(weather_data)
    st.write("Meta Data:")

if st.sidebar.button("Explore Yield Dataset"):
    yield_data = load_yield_data()
    st.write("Yield Data of various crops across farms (Displaying first 500 rows)")
    st.write("Data source: https://opendata.cbs.nl/portal.html?_la=en&_catalog=CBS&tableId=80783eng&_theme=963")
    st.write(yield_data)
    st.write("Meta Data:")

st.sidebar.subheader("EDA")
weather_yield_data = load_weather_yield_data()
regions = weather_yield_data.NAME.unique()
chosen_region = st.sidebar.selectbox(
    "Choose a region first",
    (regions)
)
chosen_crop = st.sidebar.selectbox(
    "Now choose a crop", \
    ("Potatoes",
    "Arable Vegetables",
    "Cereals",
    "Industrial Crops",
    "Pulses",
    "Sugar Beets",
    "Other Arable Crops",
    "Fruits",
    "Mushrooms")
)
if st.sidebar.button("Click to see yield"):
    st.write("Take a look at the yield details of ", chosen_crop, " in the region ", chosen_region)
    eda_year_yield_scatter(weather_yield_data, chosen_region, chosen_crop)

st.sidebar.subheader("Prediction")

show_prediction_for = st.sidebar.selectbox(
    "Choose a crop", \
    ("Potatoes",
    "Arable Vegetables",
    "Cereals",
    "Industrial Crops",
    "Pulses",
    "Sugar Beets",
    "Other Arable Crops",
    "Fruits",
    "Mushrooms")
)

if st.sidebar.button("Click to see Prediction"):
    st.write("Take a look at the PREDICTED YIELD details of ", show_prediction_for)
    st.title(predict_yield())