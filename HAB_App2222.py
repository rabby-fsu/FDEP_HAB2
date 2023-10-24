import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle


import base64 




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

feature_names = X_train.columns.tolist()

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
def predict_chla_n(input_data, model):
    predictions = model.predict(input_data)
    return predictions

# Save the trained model to a file using pickle
with open("rf_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)
    

# Function to predict Chl-a for user data (multiple as csv)
def predict_chla_n_multiple(user_data, model):
    # Ensure that the user data columns match the model's input features
    expected_columns = ['secchi', 'do_mg_l_s', 'temp_s', 'sal_ppt_s', 'turb_s', 'N/P']
    if not set(expected_columns).issubset(user_data.columns):
        st.error("The uploaded CSV file does not have the expected columns.")
        return None

    # Make predictions using the loaded model
    predictions = model.predict(user_data[expected_columns])

    return predictions


import streamlit as st

# Set a background image
background_image = "ImageApala.png"
background_css = f"""
<style>
body {{
    background-image: url("{background_image}");
    background-size: cover;
}}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)


# Add your content here


st.sidebar.image("ImageApala.png", use_column_width=True)
st.sidebar.header("Chlorophyll-a Estimation Tool for Bay-Estuary")
st.sidebar.markdown("This tool was developed training Machine Learning models to  allow user to estimate Chl-a levels in bay estuaries based on physical water quality parameters with or without meteorological parameters")



# Streamlit app
#st.sidebar.image()
#st.sidebar.image()
#st.title("Bay_Estuary_Chl_A")

# Create two options for the user to choose estimation method
estimation_method = st.radio('Estimate Chl-a with:', ['Physical Water Quality Parameters', 'Physical Water Quality & Meteorological Parameters'])



if estimation_method == 'Physical Water Quality Parameters':
    st.write("Please provide Physical Water Quality Parameters:")
    # Create an empty table for user input
    input_data = []
    for feature_name in feature_names:
        user_input = st.text_input(f"Enter {feature_name}:", key=feature_name)
        input_data.append(user_input)
        
    if st.button("Estimate Chl-a"):
        # Convert user input to a DataFrame
        user_input_data = pd.DataFrame({feature_names[i]: [input_data[i]] for i in range(len(feature_names))})
        # After receiving input data, load the model
        with open("rf_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)
        # Call the prediction function
        predictions = predict_chla_n(user_input_data, loaded_model)
        # Display predictions to the user
        st.write("Predictions for entered data:")
        st.write(predictions)        




    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        user_data = pd.read_csv(uploaded_file)
        # Make predictions using the loaded model
        with open("rf_model.pkl", "rb") as model_file:
             loaded_model = pickle.load(model_file)
        predictions = predict_chla_n_multiple(user_data, loaded_model)
        # Add a new column for estimated Chl-a
        user_data['Estimated_Chl-a'] = predictions
        # Display predictions to the user
        st.write("Predictions for uploaded data:")
        st.write(user_data)

        # Option to download the CSV file with estimated Chl-a
        csv_file_with_predictions = user_data.to_csv(index=False)
        b64 = base64.b64encode(csv_file_with_predictions.encode()).decode()
        st.markdown(f'**[Download CSV with Estimated Chl-a](data:file/csv;base64,{b64})**')


elif estimation_method == 'Meteorological Parameters':
    st.write("Please provide Meteorological Parameters:")
    # Create an empty table for user input
    input_data = []
    for feature_name in feature_names:
        user_input = st.text_input(f"Enter {feature_name}:", key=feature_name)
        input_data.append(user_input)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        user_data = pd.read_csv(uploaded_file)
        # Make predictions using the loaded model
        predictions = predict_chla_n(user_data, loaded_model)
        # Add a new column for estimated Chl-a
        user_data['Estimated_Chl-a'] = predictions
        # Display predictions to the user
        st.write("Predictions for uploaded data:")
        st.write(user_data)

        # Option to download the CSV file with estimated Chl-a
        csv_file_with_predictions = user_data.to_csv(index=False)
        b64 = base64.b64encode(csv_file_with_predictions.encode()).decode()
        st.markdown(f'**[Download CSV with Estimated Chl-a](data:file/csv;base64,{b64})**')
        


    if st.button("Estimate Chl-a"):
        # Convert user input to a DataFrame
        user_input_data = pd.DataFrame({feature_names[i]: [input_data[i]] for i in range(len(feature_names))})
        # After receiving input data, load the model
        with open("rf_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)
        # Call the prediction function
        predictions = predict_chla_n(user_input_data, loaded_model)
        # Display predictions to the user
        st.write("Predictions for entered data:")
        st.write(predictions)

# Optionally, you can add visualizations and additional information her









