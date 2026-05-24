import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained Logistic Regression model
logistic_model = joblib.load('performance_logistic_regression_model.joblib')

# Define the exact columns the model was trained on
# This list is crucial to ensure the input data matches the model's expectations
model_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
    'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
    'school_MS', 'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T',
    'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
    'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
    'reason_home', 'reason_other', 'reason_reputation', 'guardian_other',
    'guardian_mother', 'schoolsup_yes', 'famsup_yes', 'paid_yes',
    'activities_yes', 'nursery_yes', 'higher_yes', 'internet_yes',
    'romantic_yes'
]

# Define information for categorical features (original values before one-hot encoding)
# This helps create the correct Streamlit widgets and then correctly preprocess the input.
categorical_map = {
    'school': ['GP', 'MS'],
    'sex': ['F', 'M'],
    'address': ['U', 'R'],
    'famsize': ['GT3', 'LE3'],
    'Pstatus': ['A', 'T'],
    'Mjob': ['at_home', 'health', 'other', 'services', 'teacher'],
    'Fjob': ['at_home', 'health', 'other', 'services', 'teacher'],
    'reason': ['course', 'home', 'other', 'reputation'],
    'guardian': ['father', 'mother', 'other'],
    'schoolsup': ['yes', 'no'],
    'famsup': ['yes', 'no'],
    'paid': ['yes', 'no'],
    'activities': ['yes', 'no'],
    'nursery': ['yes', 'no'],
    'higher': ['yes', 'no'],
    'internet': ['yes', 'no'],
    'romantic': ['yes', 'no']
}

# Define information for numerical features (min/max/default for st.number_input)
numerical_info = {
    'age': {'min': 15, 'max': 22, 'default': 17},
    'Medu': {'min': 0, 'max': 4, 'default': 2},
    'Fedu': {'min': 0, 'max': 4, 'default': 2},
    'traveltime': {'min': 1, 'max': 4, 'default': 2},
    'studytime': {'min': 1, 'max': 4, 'default': 2},
    'failures': {'min': 0, 'max': 3, 'default': 0},
    'famrel': {'min': 1, 'max': 5, 'default': 4},
    'freetime': {'min': 1, 'max': 5, 'default': 3},
    'goout': {'min': 1, 'max': 5, 'default': 3},
    'Dalc': {'min': 1, 'max': 5, 'default': 1},
    'Walc': {'min': 1, 'max': 5, 'default': 2},
    'health': {'min': 1, 'max': 5, 'default': 3},
    'absences': {'min': 0, 'max': 93, 'default': 4},
    'G1': {'min': 0, 'max': 20, 'default': 10},
    'G2': {'min': 0, 'max': 20, 'default': 10},
}

def preprocess_input(input_data: dict, model_features: list) -> pd.DataFrame:
    """
    Preprocesses the raw input data from Streamlit to match the model's expected format.
    Applies one-hot encoding and reindexes the DataFrame.
    """
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to categorical columns, dropping the first category
    # This mimics the pd.get_dummies(df, columns=categorical_cols, drop_first=True) done during training.
    encoded_input_df = pd.get_dummies(input_df, columns=categorical_map.keys(), drop_first=True)

    # Reindex the DataFrame to match the model's training features exactly
    # Fill missing columns (categories not present in this single input row) with 0
    final_input = encoded_input_df.reindex(columns=model_features, fill_value=0)
    
    # Ensure all boolean columns are converted to the correct numerical type (0 or 1)
    for col in final_input.columns:
        if final_input[col].dtype == 'bool':
            final_input[col] = final_input[col].astype(int)

    return final_input

# --- Streamlit App Layout ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")
st.markdown("This app predicts if a student will 'Pass' (G3 >= 10) or 'Fail' (G3 < 10) based on various factors.")

st.sidebar.header("Student Information")

# Collect user input
input_data = {}

# Numerical Inputs
st.sidebar.subheader("Grades and Demographics")
for feature, info in numerical_info.items():
    if feature not in ['G1', 'G2']: # G1 and G2 are input features for the model
        input_data[feature] = st.sidebar.number_input(
            f"{feature.replace('_', ' ').title()}",
            min_value=info['min'],
            max_value=info['max'],
            value=info['default']
        )

input_data['G1'] = st.sidebar.number_input(
    "First Period Grade (G1)",
    min_value=numerical_info['G1']['min'],
    max_value=numerical_info['G1']['max'],
    value=numerical_info['G1']['default']
)
input_data['G2'] = st.sidebar.number_input(
    "Second Period Grade (G2)",
    min_value=numerical_info['G2']['min'],
    max_value=numerical_info['G2']['max'],
    value=numerical_info['G2']['default']
)

st.sidebar.subheader("Categorical Factors")
for feature, options in categorical_map.items():
    input_data[feature] = st.sidebar.selectbox(
        f"{feature.replace('_', ' ').title()}",
        options=options,
        index=0 # Default to the first option
    )

# Prediction button
if st.button("Predict Final Grade Status"):
    # Preprocess the input data
    processed_input = preprocess_input(input_data, model_features)

    # Make prediction
    prediction = logistic_model.predict(processed_input)[0]
    prediction_proba = logistic_model.predict_proba(processed_input)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The student is predicted to **PASS** with a probability of {prediction_proba[1]:.2f}")
    else:
        st.error(f"The student is predicted to **FAIL** with a probability of {prediction_proba[0]:.2f}")
    
    st.write("--- Debug Information ---")
    st.write("Processed Input for Model:")
    st.dataframe(processed_input)
    st.write(f"Model Prediction Raw Output: {prediction}")
    st.write(f"Prediction Probabilities (Fail, Pass): {prediction_proba}")
