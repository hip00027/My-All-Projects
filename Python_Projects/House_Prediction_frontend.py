import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"D:\Naresh i Class\Sept 2024\9 Sep 24\House Prediction\linear_regression_model_house.pkl", 'rb'))
#loads the pre-trained linear regression model from a file (linear_regression_model.pkl) using pickle.load().

# Set the title and header with a more appealing introduction
st.title("🏡 House Price Prediction App")
st.markdown("""
Welcome to the **House Price Prediction App**! Use this tool to estimate the market price of a house based on its living space. This app leverages advanced machine learning techniques to provide you with quick, data-driven insights.
""")

# Organize content using columns
col1, col2 = st.columns(2)

with col1:
    st.image("https://image.shutterstock.com/image-vector/real-estate-vector-icon-home-260nw-1984526906.jpg", width=150)
    
with col2:
    st.write("""
    ### How It Works:
    - 🏠 **Input**: Enter the square footage of the house you are interested in.
    - 📊 **Prediction**: The app uses a trained linear regression model to predict the price based on historical data.
    - 💡 **Output**: Instantly receive an estimated price to help you make informed decisions about buying, selling, or valuing properties.
    """)

# Add an input widget with a label and adjust default and range values
st.markdown("## Enter House Details:")
area = st.number_input("Enter Area (Square Feet):", min_value=0.0, max_value=6300.0, value=1000.0, step=50.0)

# Add prediction button with additional styling
if st.button("📈 Predict House Price"):
    # Make a prediction using the trained model
    area_input = np.array([[area]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(area_input)
    
    # Display the result with a success message and formatting
    st.success(f"💰 The predicted price of the house for {area} Sq Ft is: **${prediction[0]:,.2f}**")

# Additional information about the model and its limitations
st.info("""
**Note:** The predictions are based on past data and should be used as estimates. Actual prices may vary due to market conditions and other influencing factors.

- The model was trained using a dataset of historical house prices and square footage.
- For the best results, use the app as one of several tools in your decision-making process.
""")

# Optional: Add a footer or sidebar for additional resources
st.sidebar.markdown("### Additional Resources")
st.sidebar.markdown("""
- 📚 [Learn More About House Pricing](https://www.example.com)
- 🔍 [Check Market Trends](https://www.example.com)
- 📬 [Contact a Real Estate Expert](mailto:info@example.com)
""")


# Summary:
# The app loads a pre-trained linear regression model using pickle.
# It allows users to input the area of house via an interactive number input widget.
# Upon clicking the "Predict House Price" button, the app predicts the expected house price based on the user's input and displays the result.
# The application is designed to be simple and user-friendly, providing quick insights into salary expectations based on years of experience using a linear regression model.