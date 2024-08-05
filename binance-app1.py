import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize Binance API
binance = ccxt.binance()
limit = 10000  # number of data points to fetch

# Popular cryptocurrency pairs and timeframes
crypto_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'BNB/FDUSD', 'XRP/USDT', 'SOL/USDT']
timeframes = ['1d', '4h', '1h', '1m']  # daily, 4-hour, 1-hour timeframes

# Calculate default end date (previous day of the current month)
def get_previous_day_of_month():
    today = datetime.now()
    first_day_of_month = today.replace(day=1)
    previous_day = first_day_of_month - timedelta(days=1)
    return previous_day.date()

# Fetch historical data
def fetch_data(symbol, timeframe):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Prepare the data
def prepare_data(df):
    df['pct_change'] = df['close'].pct_change()
    df.dropna(inplace=True)
    
    # Feature engineering: use past returns to predict future returns
    df['lag1'] = df['pct_change'].shift(1)
    df['lag2'] = df['pct_change'].shift(2)
    df.dropna(inplace=True)
    
    X = df[['lag1', 'lag2']]
    y = df['pct_change']
    
    return X, y

# Train a model and make predictions
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_future(model, X_test):
    return model.predict(X_test)

def forecast_future_prices(df, model, forecast_days):
    last_known_data = df[['pct_change', 'lag1', 'lag2']].iloc[-1:]
    future_data = []

    for i in range(forecast_days):
        lag1, lag2 = last_known_data[['lag1', 'lag2']].values[0]
        pct_change = model.predict([[lag1, lag2]])[0]
        future_data.append(pct_change)
        
        # Update the last_known_data with new lags
        last_known_data = pd.DataFrame({'pct_change': [pct_change], 'lag1': [pct_change], 'lag2': [lag1]})
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    future_df = pd.DataFrame(future_data, columns=['pct_change'], index=future_dates)
    future_df['close'] = df['close'].iloc[-1] * (1 + future_df['pct_change']).cumprod()
    future_df['low'] = df['low'].iloc[-1] * (1 + future_df['pct_change']).cumprod()
    future_df['high'] = df['high'].iloc[-1] * (1 + future_df['pct_change']).cumprod()
    future_df['open'] = df['open'].iloc[-1] * (1 + future_df['pct_change']).cumprod()
    return future_df

# Streamlit App
st.title("Cryptocurrency Price Prediction with Random Forest")

# Select Box for Cryptocurrency Pairs
selected_pair = st.selectbox("Select Cryptocurrency Pair", crypto_pairs)

# Select Box for Timeframe
selected_timeframe = st.selectbox("Select Timeframe", timeframes)

# Date Range Picker
default_end_date = get_previous_day_of_month()
start_date = st.date_input("Start Date", pd.to_datetime('2023-01-01'))
end_date = st.date_input("End Date", default_end_date)
forecast_days = st.number_input("Number of Days to Forecast", min_value=1, max_value=365, value=30)

if start_date < end_date:
    # Fetch and filter data based on selected pair and timeframe
    df = fetch_data(selected_pair, selected_timeframe)
    df = df.loc[start_date:end_date]
    
    if len(df) > 2:  # Ensure enough data for feature engineering
        model_built = False
        
        # Build Model Button
        if st.button('Build Model'):
            X, y = prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train model
            model = train_model(X_train, y_train)
            
            # Make predictions
            predictions = predict_future(model, X_test)
            
            # Evaluate the model
            mse = mean_squared_error(y_test, predictions)
            st.write(f'Mean Squared Error: {mse}')
            
            # Display results
            results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)
            st.write(results.head())
            
            # Plot Actual vs Predicted Returns
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual'))
            fig1.add_trace(go.Scatter(x=results.index, y=results['Predicted'], mode='lines', name='Predicted', line=dict(dash='dash')))
            
            fig1.update_layout(title='Actual vs Predicted Returns',
                               xaxis_title='Date',
                               yaxis_title='Percentage Change',
                               template='plotly_dark')
            
            st.plotly_chart(fig1)
            
            # Forecast future prices
            future_df = forecast_future_prices(df, model, forecast_days)
            
            # Limit historical data to the number of days in forecast
            historical_limit_date = df.index[-1] - pd.Timedelta(days=forecast_days)
            historical_df = df[df.index >= historical_limit_date]
            
            # Plot Historical and Forecasted Prices
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=historical_df.index, y=historical_df['close'], mode='lines' , name='Historical ' + selected_pair))
            fig2.add_trace(go.Scatter(x=historical_df.index, y=historical_df['low'], mode='lines' , name='low_Historical ' + selected_pair))
            fig2.add_trace(go.Scatter(x=historical_df.index, y=historical_df['high'], mode='lines' , name='high_Hist ' + selected_pair))
            fig2.add_trace(go.Scatter(x=future_df.index, y=future_df['close'], mode='lines', name='Forecasted ' + selected_pair, line=dict(dash='dash')))
            fig2.add_trace(go.Scatter(x=future_df.index, y=future_df['low'], mode='lines', name='low_Forecast ' + selected_pair, line=dict(dash='dash', color='red')))
            fig2.add_trace(go.Scatter(x=future_df.index, y=future_df['high'], mode='lines', name='high_Forecast ' + selected_pair, line=dict(dash='dash', color='green')))

            fig2.update_layout(title='Historical and Forecasted ' + selected_pair + ' Prices',
                               xaxis_title='Date',
                               yaxis_title='Price (USDT)',
                               template='plotly_dark')
            
            st.plotly_chart(fig2)



            #candle stick
            st.title("Stock Market Candlestick Chart")





            # Create the candlestick chart
            fig3 = go.Figure()
            fig3.add_trace(go.Candlestick(
                x=historical_df.index,
                open=historical_df['open'],
                high=historical_df['high'],
                low=historical_df['low'],
                close=historical_df['close'],
                name='Historical'
            ))

            fig3.add_trace(go.Candlestick(
                x=future_df.index,
                open=future_df['open'],
                high=future_df['high'],
                low=future_df['low'],
                close=future_df['close'],
                name='Predicted',
                increasing=dict(line=dict(color='green', width=2), fillcolor='rgba(0, 255, 0, 0.2)'),
                decreasing=dict(line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 0.2)')
            ))

            fig3.add_trace(go.Scatter(x=future_df.index, y=future_df['close'], mode='lines', name='Forecasted ' + selected_pair, line=dict(dash='dash')))
            

            fig3.update_layout(
                title='Stock Market Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                xaxis_rangeslider_visible=False  # Hide range slider
            )

            # Display the candlestick chart in Streamlit
            st.plotly_chart(fig3)


            
            model_built = True
            
        if not model_built:
            st.write("Click the 'Build Model' button to train the model and generate predictions.")
    else:
        st.warning("Not enough data for feature engineering and model training.")
else:
    st.error("End date must be after start date.")
