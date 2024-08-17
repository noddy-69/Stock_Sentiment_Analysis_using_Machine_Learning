Stock Sentiment Analysis Using Machine Learning
Overview
The stock market is a complex, nonlinear, and dynamic system influenced significantly by investor sentiment. Research has shown that investor sentiment can drive market behavior, as highlighted by Robert J. Shiller in his analysis of the 1987 stock market crash (commonly known as Black Monday), which was largely influenced by investor actions. Quantifying investor sentiment is crucial for predicting market trends, and this project leverages sentiment analysis to extract viewpoints and attitudes from textual data. By analyzing these sentiments, we aim to predict bullish or bearish trends in the stock market, providing actionable insights to guide investment decisions.

Project Workflow
The project involves training machine learning models using a dataset spanning 2012-2021 and applying the trained model to predict a trading strategy for 2022-2023. The detailed workflow is as follows:

1. Data Collection
News Data (2012-2021):
Scraped top 10 news headlines daily from takemeback.to and CNN using BeautifulSoup.
Combined news from both sources to reduce null values in the dataset.
Financial news articles and reports were scraped from the New York Times using Selenium.
Python’s datetime module was used to iterate over each day during the period.
2. Sentiment Analysis and Feature Extraction
Cleaned the dataset by removing punctuation and processed the data using Hugging Face’s RoBERTa model to calculate sentiment scores (positive, neutral, and negative).
Calculated final sentiment scores for each day by averaging the sentiment scores of all news headlines for that day.
Applied the day-of-the-week effect for adjusting sentiment scores, acknowledging that a significant volume of news is released over the weekend, impacting Monday’s investor behavior:
Smodified = e^(-2) * S(Saturday) + e^(-1) * S(Sunday) + S(Monday)
Created lagged features for sentiment scores over the past 7 days.
Calculated 7-day rolling averages for sentiment scores.
3. Model Training
Historical Data (2012-2021): Acquired Amazon stock data from Yahoo Finance.
Created labels based on the following criteria:
Label = 0 if Close_n < Close_(n-1)
Label = 1 if Close_n > Close_(n-1)
Where Close_n is the closing price for the nth day and Close_(n-1) is the closing price for the (n-1)th day.
Generated lagged features and moving averages for closing prices similar to the sentiment scores.
Merged sentiment-based features with historical stock data.
Selected only lagged features and moving averages for model training with the target variable as the label.
Split the data into an 80-20 ratio for training and testing machine learning models.
Trained classification models including Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Classifier, and Linear Discriminant Analysis.
4. Model Evaluation
Evaluated models using the following metrics:
Accuracy
Precision
Recall
F1-Score
ROC Curve Analysis
5. Portfolio Calculation
Prepared a new dataset by scraping data for the period 2022-2023.
Preprocessed the dataset as done for the training data and used the selected model to predict the target variable.
Developed a trading strategy, identifying buy and sell points by combining the strategy and model predictions.
Executed trades and calculated the final portfolio metrics:
Sharpe Ratio: (Rp - Rf) / σ
Maximum Drawdown: ((trough value - peak value) / peak value) x 100
Win Ratio: (number of winning trades / total number of trades) x 100
Trading Strategies
1. Overnight Trading
Adopted the "Overnight Trading" strategy, where trades are placed after the market closes and before it opens. This approach is an extension of after-hours trading.
The model predicts market movement based on lagged features and moving averages for sentiment scores and stock closing prices.
2. Contrarian Trading
Applied the "Contrarian Trading" strategy, which capitalizes on market overreactions to news and events. This strategy involves:
Selling when the model predicts a market rise and buying when the model predicts a decline.
Aggressively trading by fully deploying available capital during buy signals and selling all holdings during sell signals.
Portfolio Analysis
Invested Amount: $30,000
Stocks in Portfolio:
Amazon (AMZN)
Google (GOOGL)
Microsoft (MSFT)
Results
The portfolio yielded an average return of 139.57%, driven by high returns from Amazon due to model training based on Amazon’s historical data. Excluding this outlier, the portfolio's returns ranged between 40% and 50%.

Limitations
Lower trading volumes during after-hours sessions can lead to price volatility.
Significant price gaps can occur between sessions, affecting trading strategies.
Contrarian trading can be risky during strong market trends driven by fundamental factors.
Aggressive trading requires precise market timing, which is challenging even for seasoned investors.
