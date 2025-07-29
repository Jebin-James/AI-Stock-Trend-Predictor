# --- 1. Import Libraries ---
# This cell imports all the necessary libraries for the project.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- 2. Define Helper Functions ---

def fetch_data(ticker='AAPL', start_date='2015-01-01', end_date='2025-01-01'):
    """
    Fetches historical stock data from Yahoo Finance for a given ticker.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data found for ticker {ticker}. Please check the symbol.")
        return None
    print("Data fetched successfully.")
    return stock_data

def preprocess_data(data):
    """
    Prepares and scales the data for the LSTM model.
    It creates sequences of data for time-series forecasting.
    """
    # We will use only the 'Close' price for this prediction model.
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data to be between 0 and 1. This helps the model converge faster.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences of data. We'll use the past `time_step` days to predict the next day.
    X, y = [], []
    time_step = 60  # Using 60 days of historical data
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape the data to be 3D for LSTM input: [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, time_step

def build_lstm_model(input_shape):
    """
    Builds and compiles a more extensive LSTM model with multiple layers.
    """
    model = Sequential()
    
    # Layer 1: LSTM layer with 100 units. `return_sequences=True` because we are stacking LSTM layers.
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Dropout for regularization to prevent overfitting
    
    # Layer 2: Another LSTM layer
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))

    # Layer 3: A final LSTM layer. `return_sequences=False` as it's the last LSTM layer.
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Layer 4: A dense layer for further processing
    model.add(Dense(units=50))

    # Layer 5: The final output layer with a single neuron for the predicted price
    model.add(Dense(units=1))
    
    print("Compiling the model...")
    # We use 'adam' optimizer and 'mean_squared_error' for regression problems.
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model compiled successfully.")
    model.summary() # Print a summary of the model architecture
    return model

# --- 3. Main Execution ---

# Define the stock ticker and date range for the data
TICKER = 'TSLA'  # Changed to Tesla for another example
START_DATE = '2015-01-01'
END_DATE = '2025-07-28' # Use a recent date for the end

# Fetch the data
data = fetch_data(TICKER, START_DATE, END_DATE)

if data is not None:
    original_data = data.copy() # Keep a copy for plotting later

    # Preprocess the data
    X, y, scaler, time_step = preprocess_data(data)

    # Split data into training and testing sets (80% train, 20% test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Build the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))

    # Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.1, # Use 10% of training data for validation
        verbose=1
    )
    print("--- Model Training Complete ---")

    # Save the trained model to a file
    MODEL_FILENAME = 'stock_predictor_extensive.h5'
    model.save(MODEL_FILENAME)
    print(f"\nModel saved to disk as {MODEL_FILENAME}")

    # --- 4. Model Evaluation ---

    # Make predictions on the test set
    print("\nMaking predictions on the test set...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled) # Un-scale the predictions back to original price scale

    # Get the actual prices for the test set
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
    
    # --- 5. Visualize Results ---

    # Plot the actual vs. predicted prices
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 8))
    
    # We need to calculate the correct index for plotting
    plot_index = original_data.index[split_index + time_step:]
    
    plt.plot(plot_index, actual_prices, color='cyan', label='Actual Stock Price')
    plt.plot(plot_index, predictions, color='red', linestyle='--', label='Predicted Stock Price')
    
    plt.title(f'{TICKER} Stock Price Prediction using LSTM', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Plot training & validation loss values from the model's history
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'], color='lime', label='Training Loss')
    plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
    plt.title('Model Loss During Training', fontsize=18)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()