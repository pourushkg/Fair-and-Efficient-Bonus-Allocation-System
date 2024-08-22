import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler  # For feature scaling (optional)
import warnings
warnings.filterwarnings("ignore")


def main():
    """
    Streamlit app for one-record prediction using a trained XGBoost model.
    """

    # Title and description
    st.markdown("<h1 style='color:red;'>Optimization of Bonus Allocation</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>This app allows you to predict 'Should_Receive_Bonus' for a single record based on features.</h2>", unsafe_allow_html=True)

    # User input fields
    winning_percentage = st.number_input("Winning Percentage (%)", min_value=0.0, max_value=100.0)
    average_bet_amount = st.number_input("Average Bet Amount")
    number_of_bonuses_received = int(st.number_input("Number of Bonuses Received"))
    amount_of_bonuses_received = st.number_input("Amount of Bonuses Received")
    revenue_from_bonuses = st.number_input("Revenue from Bonuses")

    # Button to trigger prediction
    if st.button("Predict"):
        # Create a DataFrame for the input data
        data = pd.DataFrame({
            "Winning_percentage": [winning_percentage],
            "Average_Bet_Amount": [average_bet_amount],
            "Number_of_Bonuses_Received": [number_of_bonuses_received],
            "Amount_of_Bonuses_Received": [amount_of_bonuses_received],
            "Revenue_from_Bonuses": [revenue_from_bonuses]
        })

        # Feature scaling (optional): If your model was trained on scaled features
        scaler = StandardScaler()  # Create a scaler object
        scaled_data = scaler.fit_transform(data)  # Scale the input data

        # Load the trained model
        #model_trainer = ModelTrainer(config=None)  # Assuming config is set elsewhere
        with open('artifacts\model_trainer\model.joblib', 'rb') as file:
            model = joblib.load(file)
        # Make prediction on the scaled data
        prediction = model.predict(scaled_data)[0]  # Get the first prediction

        # Display the prediction
        if prediction == 1:
            st.success("Prediction: User Should Receive Bonus")
        else:
            st.warning("Prediction: User Should Not Receive Bonus")

if __name__ == "__main__":
    main()