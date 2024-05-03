import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    return load_model('lstm_model.h5')

# Load the preprocessed dataset
@st.cache
def load_preprocessed_data():
    return pd.read_csv('data.csv')

# Reshape data for LSTM input
def reshape_for_lstm(data, look_back=30):
    dataset = data['Global_active_power'].values.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y), scaler

# Make predictions
def make_predictions(model, X_test, scaler):
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Main function
def main():
    # Load model
    model = load_lstm_model()
    
    # Load preprocessed data
    data = load_preprocessed_data()

    # Show title
    st.title('Global Active Power Prediction')

    # Show dataset
    st.subheader('Dataset')
    st.write(data.head())

    # Show input options
    st.sidebar.header('Input Options')

    # Show date range selector
    st.sidebar.subheader('Select Date Range')
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')

    # Perform predictions
    if st.sidebar.button('Predict'):
        # Filter data based on selected date range
        filtered_data = data[(data['date_time'] >= start_date) & (data['date_time'] <= end_date)]
        
        # Get test data for selected date range
        test_data = filtered_data['Global_active_power'].values
        
        # Reshape test data for LSTM input
        X_test, _, scaler = reshape_for_lstm(filtered_data)
        
        # Make predictions
        predictions = make_predictions(model, X_test, scaler)
        
        # Show predictions
        st.subheader('Predictions')
        st.write(predictions)

# Run the app
if __name__ == '__main__':
    main()
