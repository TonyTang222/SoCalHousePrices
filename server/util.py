import pickle
import json
import numpy as np
import pandas as pd

__locations = None
__data_columns = None
__model = None

def predict_price(location, bed, bath, sqft):
    # First, get the feature names exactly as they are expected by the model
    feature_names = __model.feature_names_in_
    
    # Create DataFrame with all necessary columns
    input_df = pd.DataFrame(columns=feature_names)
    
    # Initialize with zeros
    input_df.loc[0] = 0
    
    # Set basic features
    input_df.loc[0, 'bed'] = bed
    input_df.loc[0, 'bath'] = bath
    input_df.loc[0, 'sqft'] = sqft
    
    # Format the input location to match what's in the model's feature_names
    # We'll do a case-insensitive search
    formatted_location = None
    location_lower = location.lower()
    
    for feature in feature_names:
        if feature.lower() == location_lower:
            formatted_location = feature
            break
    
    # Set the location feature if found
    if formatted_location and formatted_location in feature_names:
        input_df.loc[0, formatted_location] = 1
    
    return round(__model.predict(input_df)[0])

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("/Users/tonytang/Desktop/SocalHousePrices/server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('/Users/tonytang/Desktop/SocalHousePrices/server/artifacts/socal_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(predict_price('San Diego, CA', 2, 2, 1500))
    print(predict_price('San Diego, CA', 3, 3, 1500))
    print(predict_price('Los Angeles, CA', 2, 2, 1500))
    print(predict_price('Los Angeles, CA', 3, 3, 1500))