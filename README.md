#Stock Predictor: Optimizing Portfolios & Predicting Trends
#Overview
This project is a comprehensive stock analysis and portfolio optimization tool, designed to provide insights into stock trends, correlations, and optimal portfolio allocations. It also includes a predictive model for forecasting stock prices using an LSTM model. The application is built using Python and integrates several powerful libraries, including TensorFlow, yfinance, and Streamlit.

#Features
Stock Data Retrieval: Automatically downloads historical stock data for specified symbols using yfinance.
Data Visualization: Visualizes closing prices, percentage changes, and moving averages (50-day, 100-day, and 200-day).
Correlation Analysis: Computes and displays a heatmap of the correlation matrix for the selected stocks.
Portfolio Optimization: Uses the Markowitz mean-variance optimization method to determine the optimal portfolio weights.
Stock Price Prediction: Utilizes a pre-trained LSTM model to predict future stock prices and evaluate model performance with RMSE.
Portfolio Evaluation: Provides insights into portfolio risk levels, expected returns, and volatility.
Comparative Analysis: Allows users to create custom portfolios and compare their performance against the optimized portfolio.
Risk-Return Tradeoff: Plots the risk-return tradeoff curve and positions the optimized portfolio within this context.



Download the Pre-trained LSTM Model:
Ensure that the LSTM model file (Stock model LSTM.keras) is placed in the specified directory.

Run the Streamlit Application:
Launch the Streamlit app by running:
streamlit run app.py

##Usage
#Inputs
Stock Symbols: Enter the stock symbols (comma-separated) you wish to analyze, e.g., GOOG,AAPL,MSFT.
Quantities: Enter the corresponding quantities for each stock (comma-separated), e.g., 10,20,15.

#Outputs
Closing Prices: Displays a DataFrame of the closing prices for the selected stocks.
Percentage Change: Plots the percentage change in stock prices over time.
Correlation Matrix: Shows a heatmap of the correlation matrix.
Portfolio Weights: Provides the optimized portfolio weights and compares them with the actual weights.
Covariance Matrix: Visualizes the covariance matrix of stock returns.
Stock Predictions: Plots the actual vs. predicted stock prices and provides buy/sell/hold recommendations.
Portfolio Value: Calculates and displays the total value of the portfolio.
Risk Level: Assesses and explains the risk level of the portfolio.
Performance Comparison: Compares the optimized portfolio with a user-defined portfolio.

#Visualizations
Closing Prices
Percentage Changes
Correlation Matrix
Covariance Matrix
Price vs. MA50, MA100, MA200
Original vs. Predicted Prices
Portfolio Allocation Pie Chart
Risk-Return Tradeoff Curve

#Recommendations
Based on the analysis and optimization results, the application provides actionable insights and recommendations for stock trading and portfolio management.

#Conclusion
This tool serves as a powerful assistant for investors looking to optimize their portfolios and predict stock trends using advanced machine learning techniques. By leveraging historical data, correlation analysis, and predictive modeling, users can make informed decisions to enhance their investment strategies.
