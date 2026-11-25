import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Train models if the saved model file does not exist
if not os.path.exists("model_comparison.pkl"):
    with st.spinner("Training machine learning models for the first time..."):
        subprocess.run(
            [sys.executable, "main.py"],
            check=True
        )

# Load trained models
comparison_data = joblib.load("model_comparison.pkl")

models = comparison_data["models"]
model_results = comparison_data["results"]
confusion_matrices = comparison_data["confusion_matrices"]

# Application title
st.title("Customer Churn Prediction Dashboard")

st.markdown(
    """
    Predict customer churn using Machine Learning.

    Enter customer information below to estimate churn probability,
    identify customer risk, and receive recommended actions.
    """
)

st.divider()
# ==========================================
# MODEL SELECTION
# ==========================================

st.subheader("Choose a Machine Learning Model")

selected_model_name = st.selectbox(
    "Select a model:",
    [
        "Select a Model",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "Gradient Boosting"
    ]
)

if selected_model_name == "Select a Model":
    st.info("Please select a machine learning model to view its results.")

else:
    # Get the selected trained model
    model = models[selected_model_name]

    st.success(f"Selected Model: {selected_model_name}")
    st.divider()
    # ==========================================
    # SELECTED MODEL PERFORMANCE
    # ==========================================

    st.subheader("Selected Model Performance")

    selected_metrics = model_results[selected_model_name]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.2f}%")
    col2.metric("Precision", f"{selected_metrics['Precision']:.2f}%")
    col3.metric("Recall", f"{selected_metrics['Recall']:.2f}%")
    col4.metric("F1-Score", f"{selected_metrics['F1-Score']:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrices[selected_model_name]

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(f"{selected_model_name} Confusion Matrix", fontsize=11)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # Overall model comparison
    st.divider()
    st.subheader("Overall Model Comparison")

    comparison_df = pd.DataFrame(model_results).T

    st.dataframe(
        comparison_df.style.format("{:.2f}"),
        use_container_width=True
    )

    st.subheader("Model Performance Comparison")

    chart_data = comparison_df[
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    ]

    st.bar_chart(chart_data, height=400)
    st.divider()

    # Best model recommendation
    # st.subheader("Best Model Recommendation")

    # best_model_name = comparison_df["F1-Score"].idxmax()

    # best_f1_score = comparison_df.loc[
    #     best_model_name, "F1-Score"
    # ]

    # st.success(f"Best Model: {best_model_name}")

    # st.metric(
    #     "Highest F1-Score",
    #     f"{best_f1_score:.2f}%"
    # )

    # st.divider()

# Customer information form
with st.form("customer_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col3:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

    st.divider()
    st.subheader("Services Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])

    with col3:
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])

    col1, col2, col3 = st.columns(3)

    with col1:
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])

    with col2:
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])

    with col3:
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.divider()
    st.subheader("Contract and Payment Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)

    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

    submit = st.form_submit_button(
    "Predict Customer Churn",
    disabled=(selected_model_name == "Select a Model")
)


# Prediction
if submit and selected_model_name != "Select a Model":
    model = models[selected_model_name]

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer_data)


    # Get prediction probabilities
    probabilities = model.predict_proba(customer_data)[0]

    # Find the probability of Churn = Yes
    classes = list(model.classes_)

    churn_index = classes.index("Yes")

    churn_probability = probabilities[churn_index] * 100

    # st.write("Model:", selected_model_name)
    # st.write("Classes:", classes)
    # st.write("Probabilities:", probabilities)
    # st.write("Churn Probability:", churn_probability)
    # Selected model risk level
    if churn_probability < 30:
        selected_risk_level = "Low Risk"

    elif churn_probability <= 60:
        selected_risk_level = "Medium Risk"

    else:
        selected_risk_level = "High Risk"

    st.session_state.prediction_history.append({
    "Tenure": tenure,
    "Contract": contract,
    "Monthly Charges": monthly_charges,
    "Model": selected_model_name,
    "Prediction": prediction[0],
    "Risk Level": selected_risk_level,
    "Churn Probability": f"{churn_probability:.2f}%"
})

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == "Yes":
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric(
    label="Churn Probability",
    value=f"{churn_probability:.2f}%"
    )

    st.progress(float(churn_probability / 100))

    st.subheader("Customer Risk Level")

    if churn_probability < 30:
        st.success("🟢 Low Risk")
    elif churn_probability <= 60:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")


    st.subheader("Recommended Action")

    if churn_probability < 30:
        st.info(
            "Maintain customer satisfaction. "
            "Continue providing good service."
        )

    elif churn_probability <= 60:
        st.warning(
            "Contact the customer to understand their needs. "
            "Consider offering a suitable promotion."
        )

    else:
        st.error(
            "Contact this customer as soon as possible. "
            "Consider a retention offer or contract upgrade."
        )

# ==========================================
# BATCH PREDICTION
# ==========================================

st.divider()
st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing multiple customer records "
    "to predict customer churn."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"],
    key="batch_csv"
)

if uploaded_file is not None:

    if selected_model_name == "Select a Model":

        st.warning("Please select a machine learning model first.")

    else:

        try:
            batch_df = pd.read_csv(uploaded_file)

            # Required features used during model training
            required_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges"
            ]

            # Check missing columns
            missing_columns = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            elif batch_df.empty:

                st.warning("The uploaded CSV file is empty.")

            else:

                # Select only the features required by the model
                batch_input = batch_df[required_columns].copy()

                # Convert numerical columns into numeric values
                numerical_columns = [
                    "SeniorCitizen",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges"
                ]

                for col in numerical_columns:
                    batch_input[col] = pd.to_numeric(
                        batch_input[col],
                        errors="coerce"
                    )

                # Get selected trained model
                batch_model = models[selected_model_name]

                # Predict customer churn
                batch_predictions = batch_model.predict(batch_input)

                # Get churn probabilities
                batch_probabilities = batch_model.predict_proba(
                    batch_input
                )

                # Find probability of Churn = Yes
                classes = list(batch_model.classes_)
                churn_index = classes.index("Yes")

                churn_probabilities = (
                    batch_probabilities[:, churn_index] * 100
                )

                # Determine risk levels
                risk_levels = []

                for probability in churn_probabilities:

                    if probability < 30:
                        risk_levels.append("Low Risk")

                    elif probability <= 60:
                        risk_levels.append("Medium Risk")

                    else:
                        risk_levels.append("High Risk")

                # Create prediction results
                result_df = batch_df.copy()

                result_df["Predicted Churn"] = batch_predictions

                result_df["Churn Probability"] = (
                    churn_probabilities.round(2)
                )

                result_df["Risk Level"] = risk_levels

                # Display results
                st.success(
                    f"Batch prediction completed using {selected_model_name}!"
                )

                st.subheader("Batch Prediction Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Customers",
                    len(result_df)
                )

                col2.metric(
                    "Predicted Churn",
                    (batch_predictions == "Yes").sum()
                )

                col3.metric(
                    "Predicted Stay",
                    (batch_predictions == "No").sum()
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # Download results
                csv = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Error processing CSV file: {e}")

st.divider()
st.subheader("Customer Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No prediction history available yet.")

    
