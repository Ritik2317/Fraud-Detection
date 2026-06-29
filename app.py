import streamlit as st
import pandas as pd
import joblib
model = joblib.load("Fraud_Detection_Pipeline.pkl")


st.title("Fraud Detection using Machine Learning")

st.markdown("Enter the transaction details below and click on the Predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type",["PAYMENT","TRANSFER","CASH_OUT","DEBIT"])
amount = st.number_input("Amount",min_value=0.0)
oldBalanceOrg = st.number_input("Old Balance (Sender)",min_value=0.0)
newBalanceOrg = st.number_input("New Balance (Sender)",min_value=0.0)
oldBalanceDest = st.number_input("Old Balance (Receiver)",min_value=0.0)
newBalanceDest = st.number_input("New Balance (Receiver)",min_value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg": oldBalanceOrg,
        "newbalanceOrig": newBalanceOrg,
        "oldbalanceDest" : oldBalanceDest,
        "newbalanceDest" : newBalanceDest
    }])

    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction : '{int(prediction)}'")

    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks okay.")