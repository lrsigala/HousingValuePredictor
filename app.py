import streamlit as st
import joblib
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Engineers new features from existing ones: `rooms_per_household`, `population_per_household`, `bedrooms_per_room`
    
    # Arguments:
        add_bedrooms_per_room, bool: defaults to True. Indicates if we want to add the feature `bedrooms_per_room`.
    """
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self  # We don't have any internal parameters. Only interested in transforming data.
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Load the pipeline
pipeline = joblib.load('/pipeline.pkl')

# Title of the app
st.title('Predict with Pipeline')
df = pd.read_csv('housing.csv')
# Display the DataFrame using Streamlit
st.write("Here is the DataFrame:")
st.dataframe(df)
#Inputs 
st.write("Introduce la información del inmueble:")
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("longitude", step=1)
    latitude = st.number_input("latitude", min_value=0, step=1)
    housing_median_age = st.number_input("housing_median_age", min_value=0, step=1)
    total_rooms = st.number_input("total_rooms", min_value=0, step=1)
with col2:
    total_bedrooms = st.number_input("total_bedrooms", min_value=0, step=1)
    population = st.number_input("population", min_value=0, step=1)
    households = st.number_input("households", min_value=0, step=1)
    median_income = st.number_input("median_income", min_value=0, step=1)

unique_ocean_proximity = df['ocean_proximity'].unique()
ocean_proximity = st.selectbox("Select ocean_proximity", unique_ocean_proximity)

# Map the input into a pandas DataFrame
new_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms':[total_rooms],
        'total_bedrooms':[total_bedrooms],
        'population':[population],
        'households':[households],
        'median_income':[median_income],
        'ocean_proximity':[ocean_proximity]
        })

# When the user clicks the predict button
if st.button('Predict'):
    # Make the prediction
    prediction = pipeline.predict(new_data)
    
    # Display the prediction
    st.write(f'The predicted class is: {prediction[0]}')
