# AI-Stock-Trend-Predictor

This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock price trends based on historical data. The system is deployed as an interactive web dashboard using Streamlit, allowing users to visualize stock data, technical indicators, and model predictions for any given stock ticker.

## 📈 Features

-   **Historical Data Analysis:** Fetches and displays up to 10 years of historical stock data.
-   **Technical Indicators:** Automatically calculates and plots 100-day and 200-day Moving Averages (MA) and the Relative Strength Index (RSI).
-   **AI-Powered Prediction:** Uses a trained LSTM model to forecast the next day's closing price.
-   **Interactive Dashboard:** A user-friendly web interface built with Streamlit to input stock tickers and view results.
-   **Model Training Script:** A Jupyter Notebook is provided to allow for retraining the model or training it on different data.

## 📂 Repository Structure

```
.
├── stock_predictor.h5        # Pre-trained Keras model file
├── app.py                      # The Streamlit web application script
├── stock_prediction_notebook.ipynb # Jupyter Notebook for model training
├── requirements.txt            # List of required Python packages
└── README.md                   # This file
```

## 🛠️ Setup and Installation

Follow these steps to set up the project environment on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/YourUsername/AI-Stock-Trend-Predictor.git](https://github.com/YourUsername/AI-Stock-Trend-Predictor.git)
cd AI-Stock-Trend-Predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

Create a `requirements.txt` file with the content below, and then run the installation command.

**`requirements.txt`:**
```
numpy
pandas
yfinance
tensorflow
scikit-learn
matplotlib
streamlit
plotly
```

**Installation Command:**
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

There are two main parts to this project: training the model and running the web app.

### Part 1: Training the LSTM Model (Optional)

If you wish to retrain the model or train it for a different stock, you can use the provided Jupyter Notebook.

1.  Ensure you have Jupyter Notebook installed (`pip install jupyter`).
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `stock_prediction_notebook.ipynb` and run the cells.
4.  This will generate a new `stock_predictor.h5` file in the root directory.

### Part 2: Running the Streamlit Web App

This is the main part of the project. Make sure the `stock_predictor.h5` file is in the same directory as `app.py`.

1.  Open your terminal or command prompt.
2.  Navigate to the project's root directory.
3.  Run the following command:
    ```bash
    streamlit run app.py
    ```
4.  Your web browser will automatically open with the application running.
5.  Enter a valid stock ticker (e.g., `GOOGL`, `MSFT`, `TSLA`) and click "Predict Trends" to see the analysis.

## 📄 Project Report

A detailed report outlining the project's introduction, abstract, tools used, steps involved, and conclusion can be found in the project deliverables.

## ⚠️ Disclaimer

This tool is for educational purposes only and should not be considered financial advice. Stock market predictions are inherently uncertain, and this model's output does not guarantee future performance.

