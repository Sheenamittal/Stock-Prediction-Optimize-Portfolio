import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


# Load the Keras model
model = tf.keras.models.load_model('D:\work\ml\stock prediction\Stock model LSTM.keras')

st.header('Stock Predictor: Optimizing Portfolios & Predicting Trends')

stocks_input = st.text_input('Enter Stock Symbols (comma-separated)', 'GOOG,AAPL,MSFT')
stocks_input = [stock.strip() for stock in stocks_input.split(',')]

quantities_input = st.text_input('Enter Quantities (comma-separated)', '10,20,15')
quantities_input = [int(quantity.strip()) for quantity in quantities_input.split(',')]

start = '2012-01-01'
end = '2024-04-06'

data = {}

for stock in stocks_input:
    data[stock] = yf.download(stock, start, end)

# Combine closing prices of all stocks into a single DataFrame
close_data = pd.concat([data[stock]['Close'] for stock in data.keys()], axis=1)
close_data.columns = data.keys()

st.subheader('Closing Prices for Selected Stocks')
st.write(close_data)

# Calculate percentage change for each stock
percentage_change = close_data.pct_change() * 100

st.subheader('Percentage Change for Selected Stocks')
plt.figure(figsize=(10, 6))
for stock in data.keys():
    plt.plot(percentage_change[stock], label=stock)

plt.title('Percentage Change of Selected Stocks')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
st.pyplot(plt)

# Labeling the graph to explain trends
st.subheader('Explanation of Trends')
for stock in data.keys():
    st.write(f'Trend for {stock}:')
    last_change = percentage_change[stock].iloc[-1]
    if last_change > 0:
        st.write('The percentage change has been increasing, indicating an upward trend.')
    elif last_change < 0:
        st.write('The percentage change has been decreasing, indicating a downward trend.')
    else:
        st.write('The percentage change has been relatively stable, indicating a sideways trend.')

# Calculate correlation matrix
correlation_matrix = close_data.corr()

# Display correlation matrix as a heatmap
st.subheader('Correlation Matrix of Selected Stocks')
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
st.pyplot(heatmap.figure)

# Explanation of correlation matrix
st.subheader('Explanation of Correlation Matrix')
for stock1 in correlation_matrix.columns:
    for stock2 in correlation_matrix.columns:
        if stock1 != stock2:
            correlation_coefficient = correlation_matrix.loc[stock1, stock2]
            if correlation_coefficient > 0.7:
                st.write(f'The correlation between {stock1} and {stock2} is high ({correlation_coefficient:.2f}), indicating a strong positive relationship.')
            elif correlation_coefficient < -0.7:
                st.write(f'The correlation between {stock1} and {stock2} is high ({correlation_coefficient:.2f}), indicating a strong negative relationship.')
            elif 0.3 < correlation_coefficient < 0.7:
                st.write(f'The correlation between {stock1} and {stock2} is moderate ({correlation_coefficient:.2f}), indicating a positive relationship.')
            elif -0.3 > correlation_coefficient > -0.7:
                st.write(f'The correlation between {stock1} and {stock2} is moderate ({correlation_coefficient:.2f}), indicating a negative relationship.')
            else:
                st.write(f'The correlation between {stock1} and {stock2} is weak ({correlation_coefficient:.2f}), indicating little to no relationship.')

# Calculate historical returns for each stock
returns = pd.DataFrame({stock: data[stock]['Close'].pct_change().dropna() for stock in data})

# Optimize portfolio
def calculate_portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, returns)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    initial_weights = np.ones(num_assets) / num_assets  # Equal weights to start with

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights equals 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Bounds for each weight

    result = minimize(negative_sharpe_ratio, initial_weights, args=(returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


def calculate_portfolio_metrics(weights, returns):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


risk_free_rate = 0.02 / 252  # Daily risk-free rate, you can change it as needed
optimized_weights = optimize_portfolio(returns, risk_free_rate)

# Display optimized and actual portfolio weights
st.subheader('Portfolio Weights')

# Create a DataFrame to display the weights
weights_data = {
    'Stock': stocks_input,
    'Optimized Weights': optimized_weights,
    'Actual Weights': quantities_input / np.sum(quantities_input)  # Normalize the actual weights
}
weights_df = pd.DataFrame(weights_data)

# Show the DataFrame
st.write(weights_df)


# Visualize covariance matrix
st.subheader('Covariance Matrix of Returns')
plt.figure(figsize=(10, 8))
covariance_heatmap = sns.heatmap(returns.cov() * 252, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
st.pyplot(covariance_heatmap.figure)
st.write("The covariance matrix represents the relationships between different stocks' returns. Positive values indicate a positive relationship, negative values indicate a negative relationship, and closer to zero indicates little to no relationship.")

# Visualize individual stock data and predictions
total_portfolio_value = 0
rmse_values = []
for stock, quantity in zip(stocks_input, quantities_input):
    st.subheader(f'Stock Data for {stock}')
    st.write(data[stock])

    data_train = pd.DataFrame(data[stock].Close[0: int(len(data[stock])*0.80)])
    data_test = pd.DataFrame(data[stock].Close[int(len(data[stock])*0.80): len(data[stock])])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader(f'Price vs MA50 for {stock}')
    ma_50_days = data[stock].Close.rolling(50).mean()
    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(ma_50_days, 'r', label='MA50')
    ax1.plot(data[stock].Close, 'g', label='Close Price')
    ax1.legend()
    st.pyplot(fig1)

    st.subheader(f'Price vs MA50 vs MA100 for {stock}')
    ma_100_days = data[stock].Close.rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.plot(ma_50_days, 'r', label='MA50')
    ax2.plot(ma_100_days, 'b', label='MA100')
    ax2.plot(data[stock].Close, 'g', label='Close Price')
    ax2.legend()
    st.pyplot(fig2)

    st.subheader(f'Price vs MA100 vs MA200 for {stock}')
    ma_200_days = data[stock].Close.rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    ax3.plot(ma_100_days, 'r', label='MA100')
    ax3.plot(ma_200_days, 'b', label='MA200')
    ax3.plot(data[stock].Close, 'g', label='Close Price')
    ax3.legend()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)
    
     # Scale the predictions back to the original price scale
    predicted_prices = predict[:, 0] * scaler.scale_

    # Scale the actual prices back to the original price scale
    actual_prices = y * scaler.scale_

   

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader(f'Original Price vs Predicted Price for {stock}')
    fig4, ax4 = plt.subplots(figsize=(8,6))
    ax4.plot(predict, 'r', label='Original Price')
    ax4.plot(y, 'g', label='Predicted Price')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    st.pyplot(fig4)

    # Recommendation based on trend
    current_price = data[stock].Close.iloc[-1]
    ma_50 = data[stock].Close.rolling(50).mean().iloc[-1]
    ma_100 = data[stock].Close.rolling(100).mean().iloc[-1]
    ma_200 = data[stock].Close.rolling(200).mean().iloc[-1]

    if current_price > ma_50 and current_price > ma_100 and current_price > ma_200:
        recommendation = "Strong Buy"
        reason = f"The stock price of {stock} is currently above its 50-day, 100-day, and 200-day moving averages, indicating a strong upward trend."
    elif current_price > ma_50 and current_price > ma_100:
        recommendation = "Buy"
        reason = f"The stock price of {stock} is currently above its 50-day and 100-day moving averages, suggesting a positive trend."
    elif current_price < ma_50 and current_price < ma_100:
        recommendation = "Sell"
        reason = f"The stock price of {stock} is currently below its 50-day and 100-day moving averages, indicating a downward trend."
    else:
        recommendation = "Hold"
        reason = f"The stock price of {stock} is currently in a consolidation phase, with no clear trend direction."

    st.subheader(f'Recommendation for {stock}')
    st.write(f"Based on current trends, it is recommended to {recommendation} the stock.")
    st.write(f"Reason: {reason}")

    # Calculate current value of the stock
    current_value = current_price * quantity
    st.write(f'Current Value of {stock}: ${current_value}')
    total_portfolio_value += current_value

     # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    rmse_values.append(rmse)

    # Display RMSE
    st.subheader(f'Model Evaluation for {stock}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    
# Display total portfolio value
st.subheader('Total Portfolio Value')
st.write(f'The total value of your portfolio is: ${total_portfolio_value}')

# Portfolio Allocation Pie Chart
allocation_percentages = [(current_price * quantity / total_portfolio_value) * 100 for stock, quantity in zip(stocks_input, quantities_input)]
fig, ax = plt.subplots()
ax.pie(allocation_percentages, labels=stocks_input, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.subheader('Portfolio Allocation')
st.pyplot(fig)
st.write("The pie chart shows the allocation of your portfolio among different stocks.")

# Calculate portfolio volatility
portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns.cov() * 252, optimized_weights)))


# Determine risk level and reason
risk_levels = {
    'Low': (0, 0.1, 'The portfolio has low volatility, indicating relatively stable performance.'),
    'Moderate': (0.1, 0.2, 'The portfolio has moderate volatility, suggesting some fluctuations but manageable risk.'),
    'High': (0.2, np.inf, 'The portfolio has high volatility, implying significant fluctuations and higher risk.')
}

risk_level = None
risk_reason = None
for level, (min_volatility, max_volatility, reason) in risk_levels.items():
    if min_volatility <= portfolio_volatility < max_volatility:
        risk_level = level
        risk_reason = reason
        break

# Add trend-based reason
if risk_reason:
    for stock in stocks_input:
        current_price = data[stock].Close.iloc[-1]
        ma_50 = data[stock].Close.rolling(50).mean().iloc[-1]
        ma_100 = data[stock].Close.rolling(100).mean().iloc[-1]
        ma_200 = data[stock].Close.rolling(200).mean().iloc[-1]

        if risk_level == 'Low':
            if current_price > ma_50 and current_price > ma_100 and current_price > ma_200:
                risk_reason += f' The trend for {stock} is strongly upward.'
            elif current_price > ma_50 and current_price > ma_100:
                risk_reason += f' The trend for {stock} is upward.'
            elif current_price < ma_50 and current_price < ma_100:
                risk_reason += f' The trend for {stock} is downward.'
            else:
                risk_reason += f' The trend for {stock} is sideways.'

        elif risk_level == 'Moderate':
            if current_price > ma_50 and current_price > ma_100 and current_price > ma_200:
                risk_reason += f' The trend for {stock} is upward with some fluctuations.'
            elif current_price > ma_50 and current_price > ma_100:
                risk_reason += f' The trend for {stock} is upward with occasional dips.'
            elif current_price < ma_50 and current_price < ma_100:
                risk_reason += f' The trend for {stock} is downward with some fluctuations.'
            else:
                risk_reason += f' The trend for {stock} is sideways with some fluctuations.'

        elif risk_level == 'High':
            if current_price > ma_50 and current_price > ma_100 and current_price > ma_200:
                risk_reason += f' The trend for {stock} is highly volatile with sharp fluctuations.'
            elif current_price > ma_50 and current_price > ma_100:
                risk_reason += f' The trend for {stock} is upward with high volatility.'
            elif current_price < ma_50 and current_price < ma_100:
                risk_reason += f' The trend for {stock} is downward with high volatility.'
            else:
                risk_reason += f' The trend for {stock} is highly volatile with no clear direction.'

# Display risk level and reason
st.subheader('Portfolio Risk Level')
st.write(f'The risk level of your portfolio is: {risk_level}')
if risk_reason:
    st.write(f'Reason: {risk_reason}')
else:
    st.write('Reason: Unable to determine the reason for this risk level.')

# Performance Evaluation
# Define Additional Portfolios
# Ask user to create another portfolio based on their choice
st.subheader('Create Another Portfolio')
st.write('Enter the weights for the stocks in your other portfolio (comma-separated)')
other_weights_input = st.text_input('Weights for Other Portfolio', '0.2,0.3,0.5')
other_portfolio_weights = [float(weight.strip()) for weight in other_weights_input.split(',')]

# Normalize weights if necessary
other_portfolio_weights = np.array(other_portfolio_weights)
other_portfolio_weights /= np.sum(other_portfolio_weights)

# Calculate Performance Metrics for the Markowitz Portfolio
markowitz_metrics = calculate_portfolio_metrics(optimized_weights, returns)


# Calculate Performance Metrics for the Other Portfolio
other_portfolio_metrics = calculate_portfolio_metrics(other_portfolio_weights, returns)

# Compare Portfolios
# Compare the performance metrics of the Markowitz portfolio with those of the other portfolio
st.subheader('Performance Comparison')
comparison_data = {
    'Metric': ['Annualized Return', 'Volatility', 'Sharpe Ratio'],
    'Markowitz Portfolio': markowitz_metrics,
    'Other Portfolio': other_portfolio_metrics
}

comparison_df = pd.DataFrame(comparison_data).set_index('Metric')
st.write(comparison_df)

# Provide Recommendations
# Based on the comparison, provide recommendations or insights
st.subheader('Recommendations')
if markowitz_metrics[2] > other_portfolio_metrics[2]:
    st.write('The Markowitz portfolio has a higher Sharpe ratio, indicating better risk-adjusted returns.')
    st.write('Recommendation: Stick with the Markowitz portfolio.')
elif markowitz_metrics[2] < other_portfolio_metrics[2]:
    st.write('The other portfolio has a higher Sharpe ratio, indicating better risk-adjusted returns.')
    st.write('Recommendation: Consider switching to the other portfolio.')
else:
    st.write('Both portfolios have similar risk-adjusted returns.')
    st.write('Recommendation: Stick with the portfolio you are comfortable with.')
    


# Calculate expected returns and volatility for different combinations of portfolio weights
expected_returns = []
volatility_values = []
weights_range = np.linspace(0, 1, 100)

for w in weights_range:
    weights = np.array([w] + [0] * (len(stocks_input) - 1))  # Adjust the shape of weights
    expected_return, volatility, _ = calculate_portfolio_metrics(weights, returns)  # Update the function to return the Sharpe ratio as well
    expected_returns.append(expected_return)
    volatility_values.append(volatility)

st.subheader('Risk-return tradeoff curve')

# Plot the risk-return tradeoff curve
plt.figure(figsize=(10, 8))
plt.plot(volatility_values, expected_returns, label='Risk-Return Tradeoff')
plt.scatter(portfolio_volatility, markowitz_metrics[0], color='red', label='Optimized Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Risk-Return Tradeoff Curve')
plt.legend()
plt.grid(True)

# Add explanation based on the position of the optimized portfolio
if markowitz_metrics[0] > max(expected_returns) or portfolio_volatility > max(volatility_values):
    explanation = "The optimized portfolio is positioned above the risk-return tradeoff curve, indicating a higher expected return compared to portfolios with similar volatility."
elif markowitz_metrics[0] < min(expected_returns) or portfolio_volatility < min(volatility_values):
    explanation = "The optimized portfolio is positioned below the risk-return tradeoff curve, indicating a lower expected return compared to portfolios with similar volatility."
else:
    explanation = "The optimized portfolio is positioned on the risk-return tradeoff curve, balancing expected return with volatility effectively."

plt.text(portfolio_volatility, markowitz_metrics[0], " Optimized Portfolio", fontsize=12, ha='right')

plt.annotate(explanation, xy=(0.5, 0.5), xycoords='axes fraction', xytext=(0.5, 0.9),
             fontsize=12, ha='center', va='center',
             arrowprops=dict(facecolor='black', shrink=0.05))

st.pyplot(plt)


