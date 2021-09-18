from pandas.core.algorithms import isin
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
	page_title = "Crop yield prediction",
	page_icon = "corn"
	)

st.title('Task 4 - Crop Yield Prediction')
with st.expander("Problem Statement"):
    st.write("""To predict the yield of a specific crops by 
        using data such as rainfall, temperature, soil composition, 
        PH-values etc. """)
with st.expander("Approach"):
    st.write(""" Multi Linear regression:  """)
with st.expander("Models used"):
    st.write("""sklearn Linear regression:  """)

@st.cache
# To start with, just loading and returning
# We can show filtered details little later
def load_province_data():
    province_data = pd.read_csv("data/Province_data.csv", delimiter=";")
    columns = province_data.columns
    print(columns)
    filtered_province_data = province_data[['Geo Point', 'Provincie name', 'Gemeente name']]
    return filtered_province_data

@st.cache
def load_weather_stations_data():
    weather_stations_data = pd.read_csv("data/weather_knmi_stations_2018.csv", delimiter=";")
    return weather_stations_data

@st.cache
def load_weather_data():
    weather_data = pd.read_csv("data/province_weather_2021 (1).csv", delimiter=";", nrows=500)
    return weather_data

@st.cache
def load_yield_data():
    yield_data = pd.read_csv("data/Yield_data_cbs.csv", delimiter=";")
    yield_data = yield_data[["FarmTypes","Periods","NumberOfFarmsTotal_1","PotatoesTotal_9", "VegetablesArable_13", "Cereals_14", "IndustrialCrops_16",
"Pulses_17", "SugarBeets_18", "OtherArableCrops_19", "FruitInTheOpen_44", "MushroomsTotal_68"]]
    yield_data = yield_data.iloc[:500]
    return yield_data

@st.cache
def load_weather_yield_data():
    weather_yield_data = pd.read_csv("data/merged_weather_yield_(3).csv")
    
    weather_yield_data_farm1 = weather_yield_data[weather_yield_data['FarmTypes']=='A009481']
    return weather_yield_data_farm1

@st.cache
def load_2021_weather():
    weather_2021 = pd.read_csv('data/province_weather_2021 (1).csv', index_col=0)
    weather_2021.drop(['STN', 'year'], axis=1, inplace=True)
    return weather_2021


def eda_year_yield_scatter(weather_yield_data, chosen_region, chosen_crop):
    # Line chart for crop-yield

    # actual_years = weather_yield_data.Year.unique()

    # Default is Potato
    internal_crop_name = "PotatoesTotal_9"

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
        internal_crop_name = "SugarBeets_18"
    elif chosen_crop == "Other Arable Crops":
        internal_crop_name = "OtherArableCrops_19"
    elif chosen_crop == "Fruits":
        internal_crop_name = "FruitInTheOpen_38"
    elif chosen_crop == "Mushrooms":
        internal_crop_name = "MushroomsTotal_68"

    weather_yield_data = weather_yield_data[weather_yield_data['Province'] == chosen_region]
    
        # (weather_yield_data['year'].isin(actual_years)) & \
        # (weather_yield_data['FarmTypes'] == 'A009481'))
    
    fig = px.scatter(x=weather_yield_data['Year'], y=weather_yield_data[internal_crop_name], labels={
                     "x": "Year",
                        "y": "Crop yield (in millions per hectare)"})
    st.plotly_chart(fig)
    # fig.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(
    #     ,
    #     ,
    # )
    # ax.set_xlabel("Year")
    # ax.set_xlim([2001, 2018])
    # ax.set_xticklabels(actual_years, rotation=45, ha="right")
    # ax.set_ylabel("Crop yield (in millions per hectare)")
    
    # st.write(fig)

def predict_yield(crop_name, province):

    weather_2021 = load_2021_weather()
    selected_province_weather = weather_2021[weather_2021['Province']== province]
    selected_province_weather.drop('Province', axis=1, inplace=True)

    if crop_name == "Potatoes":
        potatoe_model =pickle.load(open('models/potatoes.sav', 'rb'))
        predict = potatoe_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Arable Vegetables":
        arable_veg_model = pickle.load(open('models/arable_vegetables.sav', 'rb'))
        predict = arable_veg_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Cereals":
        cereal_model = pickle.load(open('models/cereal_model.sav', 'rb'))
        predict = cereal_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Industrial Crops":
        industrial_model= pickle.load(open('models/industrial_crops_model.sav', 'rb'))
        predict = industrial_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Pulses":
        pulses_model = pickle.load(open('models/pulses_model.sav', 'rb'))
        predict = pulses_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Sugar Beets":
        sugar_model = pickle.load(open('models/sugar_beets_model.sav', 'rb'))
        predict = sugar_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Other Arable Crops":
        other_arable_model = pickle.load(open('models/other_arable_crops_model.sav', 'rb'))
        predict = other_arable_model.predict(selected_province_weather)
        return predict

    elif crop_name == "Fruits":
        fruits_model = pickle.load(open('models/fruits_model.sav', 'rb'))
        predict = fruits_model.predict(selected_province_weather)
        return predict 
 

# Explore each dataset that is used in this task
st.sidebar.subheader("Datasets Used")
if st.sidebar.button("Explore Provinces Dataset"):
    province_data = load_province_data()
    st.write("Quick look of Provinces data set, more from a geographical location perspective")
    st.write("Data source: https://github.com/MHaringa/spatialrisk")
    st.write(province_data)
    st.write("""Meta Data: 

    Provinsie name         Province name

    Gemeente name          Municipalities name""")

if st.sidebar.button("Explore Weather Stations Dataset"):
    weather_stations_data = load_weather_stations_data()
    st.write("Weather stations in Netherlands (2018 data)")
    st.write("Data source: https://www.kaggle.com/sinaasappel/historical-weather-in-the-netherlands-19012018 ")
    st.write(weather_stations_data)
    st.write("""Meta Data: 
    STN        Station number

    LON        Longitude

    LAT        Latitude

    ALT        Altitude

    NAME       Station name
""")

if st.sidebar.button("Explore Weather Dataset"):
    weather_data = load_weather_data()
    st.write("Historical weather in the Netherlands collected from 1901 to 2018 (Displaying first 500 records)")
    st.write("Data source: https://www.kaggle.com/sinaasappel/historical-weather-in-the-netherlands-19012018 ")
    st.write(weather_data)
    st.write("""Meta Data: 

DDVEC     = Vector mean wind direction in degrees (360=north, 90=east, 180=south, 270=west, 0=calm/variable)

FHVEC     =  Vector mean windspeed (in 0.1 m/s)

FG        = Daily mean windspeed (in 0.1 m/s)

FHX       = Maximum hourly mean windspeed (in 0.1 m/s)

FHXH      = Hourly division in which FHX was measured

FHN       = Minimum hourly mean windspeed (in 0.1 m/s)

FHNH      = Hourly division in which FHN was measured

FXX       = Maximum wind gust (in 0.1 m/s)

FXXH      = Hourly division in which FXX was measured

TG        = Daily mean temperature in (0.1 degrees Celsius)

TN        = Minimum temperature (in 0.1 degrees Celsius)

TNH       = Hourly division in which TN was measured

TX        = Maximum temperature (in 0.1 degrees Celsius)

TXH       = Hourly division in which TX was measured

T10N      = Minimum temperature at 10 cm above surface (in 0.1 degrees Celsius)

T10NH     = 6-hourly division in which T10N was measured; 6=0-6 UT, 12=6-12 UT, 18=12-18 UT, 24=18-24 UT

SQ        = Sunshine duration (in 0.1 hour) calculated from global radiation (-1 for <0.05 hour)

SP        = Percentage of maximum potential sunshine duration

Q         = Global radiation (in J/cm2)

DR        = Precipitation duration (in 0.1 hour)

RH        = Daily precipitation amount (in 0.1 mm) (-1 for <0.05 mm)

RHX       = Maximum hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)

RHXH      = Hourly division in which RHX was measured

PG        = Daily mean sea level pressure (in 0.1 hPa) calculated from 24 hourly values

PX        = Maximum hourly sea level pressure (in 0.1 hPa)

PXH       = Hourly division in which PX was measured

PN        = Minimum hourly sea level pressure (in 0.1 hPa)

PNH       = Hourly division in which PN was measured

VVN       = Minimum visibility; 0: <100 m, 1:100-200 m, 2:200-300 m,..., 49:4900-5000 m, 50:5-6 km, 56:6-7 km, 57:7-8 km,..., 79:29-30 km, 80:30-35 km, 81:35-40 km,..., 89: >70 km)

VVNH      = Hourly division in which VVN was measured

VVX       = Maximum visibility; 0: <100 m, 1:100-200 m, 2:200-300 m,..., 49:4900-5000 m, 50:5-6 km, 56:6-7 km, 57:7-8 km,..., 79:29-30 km, 80:30-35 km, 81:35-40 km,..., 89: >70 km)

VVXH      = Hourly division in which VVX was measured

NG        = Mean daily cloud cover (in octants, 9=sky invisible)

UG        = Daily mean relative atmospheric humidity (in percents)

UX        = Maximum relative atmospheric humidity (in percents)

UXH       = Hourly division in which UX was measured

UN        = Minimum relative atmospheric humidity (in percents)

UNH       = Hourly division in which UN was measured

EV24      = Potential evapotranspiration (Makkink) (in 0.1 mm)""")

if st.sidebar.button("Explore Yield Dataset"):
    yield_data = load_yield_data()
    st.write("Yield Data of various crops across farms (Displaying first 500 rows)")
    st.write("Data source: https://opendata.cbs.nl/portal.html?_la=en&_catalog=CBS&tableId=80783eng&_theme=963")
    st.write(yield_data)
    st.write()

st.sidebar.subheader("EDA")
weather_yield_data = load_weather_yield_data()
regions = weather_yield_data.Province.unique()
chosen_region = st.sidebar.selectbox(
    "Choose a region first",
    regions, key= 'select1'
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

weather_2021 = load_2021_weather()

regions_2021 = weather_2021.Province.unique()
chosen_2021 = st.sidebar.selectbox(
    "Choose a region first",
    (regions_2021)
)


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
    st.write(predict_yield(show_prediction_for, chosen_2021)[0], "million tons per hectare" )