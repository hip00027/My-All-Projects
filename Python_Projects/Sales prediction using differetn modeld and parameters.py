import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import streamlit as st

# Load the dataset
dataset = pd.read_csv(r"D:\Naresh i Class\Sept 2024\19 Sep 24\Comparison of models\Sales_price_pred.csv")

# Clean up column names to remove any extra spaces
dataset.columns = dataset.columns.str.strip()

# Step 1: Handling Missing Values using SimpleImputer
# Initialize SimpleImputer with the strategy 'mean'
imputer = SimpleImputer(strategy='mean')

# Apply imputer on columns that have numerical missing values (TV, Radio, Social Media, Sales)
dataset[['TV', 'Radio', 'Social Media', 'Sales']] = imputer.fit_transform(dataset[['TV', 'Radio', 'Social Media', 'Sales']])

# Step 2: Encoding Categorical Data
# 'Influencer' column is categorical; encode it
label_encoder = LabelEncoder()

# Print unique values before encoding
print("Unique Influencer Categories Before Encoding:", dataset['Influencer'].unique())

# Fit the encoder and transform the data
dataset['Influencer'] = label_encoder.fit_transform(dataset['Influencer'])

# Print the mapping of influencer types to their encoded values to verify that the categories in the dataset were encoded correctly and to ensure there are no inconsistencies or missing values in the encoding process
print("Influencer Encoding Mapping:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# Step 3: Splitting the dataset into Independent (X) and Dependent (y) variables
X = dataset[['TV', 'Radio', 'Social Media', 'Influencer']].values
y = dataset['Sales'].values
# .values extracts the underlying data from these columns as a NumPy array, which is a common format for machine learning algorithms in Python.
# X will be a 2D array where each row represents one observation (a combination of TV, Radio, Social Media, and Influencer investments) and each column represents a feature (TV, Radio, etc.).

# Title for the Streamlit app
st.title("Interactive Sales Prediction Model Tuning")

# Sidebar for inputs
tv = st.number_input('TV Advertising Budget', min_value=0.0)
radio = st.number_input('Radio Advertising Budget', min_value=0.0)
social_media = st.number_input('Social Media Advertising Budget', min_value=0.0)
influencer_type = st.selectbox('Influencer Type', options=label_encoder.classes_)
influencer_encoded = label_encoder.transform([influencer_type])[0]

# Combine inputs for prediction
input_data = np.array([[tv, radio, social_media, influencer_encoded]])

# Step 4: Model Selection by User (Using full names directly in the dropdown)
model_choice = st.sidebar.selectbox("Choose a model", 
                                    options=["Linear Regression", 
                                             "Polynomial Regression", 
                                             "Support Vector Regression", 
                                             "K-Nearest Neighbors", 
                                             "Decision Tree Regression", 
                                             "Random Forest Regression"])


# Map the model choices to the actual algorithms
if model_choice == "Linear Regression":
    st.sidebar.write("No hyperparameters to tune for Linear Regression.")
    model = LinearRegression()

elif model_choice == "Polynomial Regression":
    degree = st.sidebar.slider("Degree of the polynomial", min_value=2, max_value=5, value=2)
    
    # Apply PolynomialFeatures transformation on the entire dataset
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    
    # Fit the Linear Regression model to the transformed dataset
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Ensure that input_data is also transformed the same way
    input_data_poly = poly_reg.transform(input_data)
    
    # Make the prediction using the transformed input data
    prediction_poly = model.predict(input_data_poly)

elif model_choice == "Support Vector Regression":
    kernel = st.sidebar.selectbox("Kernel type", ["linear", "poly", "rbf"])
    degree = st.sidebar.slider("Degree (for poly kernel)", min_value=2, max_value=5, value=3)
    model = SVR(kernel=kernel, degree=degree if kernel == "poly" else 3)

elif model_choice == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of neighbors", min_value=1, max_value=20, value=5)
    weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

elif model_choice == "Decision Tree Regression":
    criterion = st.sidebar.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error"])
    splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
    model = DecisionTreeRegressor(criterion=criterion, splitter=splitter)

elif model_choice == "Random Forest Regression":
    n_estimators = st.sidebar.slider("Number of trees in the forest", min_value=10, max_value=100, value=35)
    criterion = st.sidebar.selectbox("Criterion", ["squared_error", "absolute_error"])
    model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
    
# Train the selected model
model.fit(X, y)

# Step 5: Make Predictions and Display the Results
st.subheader("Model Prediction")

if st.button("Predict"):
    # If polynomial regression is chosen, use the transformed input data
    if model_choice == "Polynomial Regression":
        prediction = prediction_poly
    else:
        prediction = model.predict(input_data)
    
    st.write(f"Predicted Sales ({model_choice}): {prediction[0]:.2f}")






