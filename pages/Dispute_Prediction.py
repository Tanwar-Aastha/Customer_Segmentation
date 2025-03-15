import streamlit as st
import os
import pandas as pd
from datetime import date
from src.pipeline.training_pipeline import PredictPipelineClass, CustomDataClass

# Load Data (Assume CSV contains "Product" and "Issue" columns)
@st.cache_data
def load_issues():
    # Get the directory where demo.py is located (which is the project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the CSV file
    csv_path = os.path.join(BASE_DIR, "..", "dataset", "modified_data.csv")
    df = pd.read_csv(csv_path)  # Ensure this CSV exists
    return df

issues_df = load_issues()

# Initialize Prediction Pipeline
predictor = PredictPipelineClass()

# Streamlit UI
st.title("Consumer Dispute Predictor")
st.write("Predict whether a consumer complaint will be disputed based on provided details.")

# User Inputs
Date_Received = st.date_input("Date Received", value=date.today())

# Select Product Name
Product_Name = st.selectbox("Product Name", ["Select"] + sorted(issues_df["Product"].unique()))

# Dynamically filter issues based on selected product
if Product_Name != "Select":
    filtered_issues = issues_df[issues_df["Product"] == Product_Name]["Issue"].unique()
    Issue_Detail = st.selectbox("Issue Detail", ["Select"] + sorted(filtered_issues))
else:
    Issue_Detail = st.selectbox("Issue Detail", ["Select"])

# Other Inputs
Submitted_Via = st.selectbox("Submitted Via", ["Select", "Email", "Web", "Phone", "Postal mail", "Referral", "Fax"])
Date_sent_to_company = st.date_input("Date Sent to Company", value=date.today())

Company_Response = st.selectbox("Company Response", ["Select",
    "Closed with explanation", "Closed with non-monetary relief", 
    "Closed with monetary relief", "Closed without relief", 
    "Closed", "Closed with relief"
])

Timely_Response = st.radio("Timely Response?", ["Yes", "No"])

# Prediction Button
if st.button("Predict"):
    if Submitted_Via == "Select" or Company_Response == "Select" or Issue_Detail == "Select":
        st.error("⚠️ Please fill in all required fields.")
    else:
        # Convert input into a DataFrame using CustomDataClass
        custom_data = CustomDataClass(Date_Received, Product_Name, Issue_Detail, Submitted_Via,
                                      Date_sent_to_company, Company_Response, Timely_Response)
        input_df = custom_data.get_data_as_dataframe()
        
        # Make Prediction using PredictPipelineClass
        prediction = predictor.predict(input_df)

        # Display Prediction
        st.subheader("Prediction Result:")
        st.success(f"The predicted dispute status: **{prediction}**")
