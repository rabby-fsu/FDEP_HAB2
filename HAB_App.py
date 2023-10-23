

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load your dataset or replace this with your data loading code
# Assuming your data is in a DataFrame called 'df'
file_path = 'Datafile_Ml_1.csv'
df = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Select the features and target variable
selected_features = ['secchi', 'do_mg_l_s', 'temp_s', 'sal_ppt_s', 'turb_s', 'N/P']
X = df[selected_features]  # Independent variables
y = df['chla_n']           # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Step 2: Train a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Function to make predictions
def predict_chla_n(input_data):
    predictions = rf_model.predict(input_data)
    return predictions

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Function to make predictions
def predict_chla_n(input_data):
    predictions = rf_model.predict(input_data)
    return predictions

# Save the trained model to a file using pickle
with open("rf_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)
    
    


# Streamlit app
#st.sidebar.image()
# Create two options for the user to choose estimation method
estimation_method = st.radio('Estimate Chl-a with:', ['Physical Water Quality Parameters', 'Meteorological Parameters'])

if estimation_method == 'Physical Water Quality Parameters':
    st.write("Please provide Physical Water Quality Parameters:")
    # Your input fields for Physical Water Quality Parameters go here
    # ...

    # After receiving input data, load the model
    with open("rf_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    
    # Once you have the user's input data, convert it into a DataFrame and call the prediction function
    # input_data = pd.DataFrame({'param1': value1, 'param2': value2, ...})
    # predictions = predict_chla_n(input_data)
    
    # Display predictions to the user
    # st.write("Predictions:")
    # st.write(predictions)

elif estimation_method == 'Meteorological Parameters':
    st.write("Please provide Meteorological Parameters:")
    # Your input fields for Meteorological Parameters go here
    # ...

    # After receiving input data, load the model
    with open("rf_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    
    # Once you have the user's input data, convert it into a DataFrame and call the prediction function
    # input_data = pd.DataFrame({'param1': value1, 'param2': value2, ...})
    # predictions = predict_chla_n(input_data)
    
    # Display predictions to the user
    # st.write("Predictions:")
    # st.write(predictions)

# Optionally, you can add visualizations and additional information here.