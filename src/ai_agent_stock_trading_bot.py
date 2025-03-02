from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import ollama
import yfinance as yf
import logging
import requests
from textblob import TextBlob
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Gather News and Perform Sentiment Analysis
def gather_news_and_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Gather news about the company and perform sentiment analysis.
    """
    try:
        news_api_key = os.getenv('NEWS_API')
        if not news_api_key:
            logger.warning("NEWS_API environment variable not set, using placeholder sentiment data")
            return {"news": [], "average_sentiment": 0.0}
            
        news_api_url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}"
        response = requests.get(news_api_url)
        news_data = response.json()
        
        if news_data["status"] != "ok":
            raise ValueError(f"Failed to fetch news data: {news_data.get('message', 'Unknown error')}")

        articles = news_data["articles"]
        sentiment_scores = []

        for article in articles:
            if article["description"] is None:
                continue
            sentiment = TextBlob(article["description"]).sentiment.polarity
            sentiment_scores.append(sentiment)
        
        average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        return {"news": articles[:5], "average_sentiment": average_sentiment}  # Limit to 5 articles for brevity
    except Exception as e:
        logger.error(f"Failed to gather news and perform sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to gather news and perform sentiment analysis: {str(e)}")

# Step 2: Analyze Market Trends using Scikit-Learn
def analyze_market_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze market trends using scikit-learn.
    This example uses a simple linear regression model to predict the trend.
    """
    try:
        # Ensure the DataFrame has the required columns
        if 'Close' not in df.columns:
            raise ValueError("The DataFrame does not contain 'Close' prices.")

        # Prepare the data
        df = df.reset_index()  # Convert the index (Date) to a column
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by="Date")
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        if len(df) < 2:
            raise ValueError("Insufficient data for trend analysis.")

        # Prepare data for linear regression
        X = df[['Days']]
        y = df['Close']

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the next day's closing price
        next_day = df['Days'].max() + 1
        predicted_price = model.predict([[next_day]])[0]  # Extract scalar value

        # Determine the trend (buy, sell, or hold)
        last_price = df['Close'].iloc[-1]
        if predicted_price > last_price * 1.01:  # 1% threshold for buy
            trend = "buy"
        elif predicted_price < last_price * 0.99:  # 1% threshold for sell
            trend = "sell"
        else:
            trend = "hold"

        return {"trend": trend, "last_price": float(last_price), "predicted_price": float(predicted_price)}
    except Exception as e:
        logger.error(f"Failed to analyze market trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze market trends: {str(e)}")

# Step 3: Use Ollama for Inference
def generate_insights_with_ollama(news_data: Dict[str, Any], market_trend: Dict[str, Any]) -> str:
    """
    Generate insights using Ollama based on the analyzed data.
    """
    try:
        # Extract only titles and descriptions from news to avoid overly large prompts
        simplified_news = []
        for article in news_data['news'][:3]:  # Limit to first 3 articles
            simplified_news.append({
                "title": article.get("title", ""),
                "description": article.get("description", "")
            })
            
        query = f"""
        Based on the following market data and news sentiment analysis, provide brief investment insights:
        - Symbol: {market_trend.get('symbol', 'Unknown')}
        - Last Price: ${market_trend['last_price']:.2f}
        - Predicted Price: ${market_trend['predicted_price']:.2f}
        - Trend: {market_trend['trend']}
        - Average News Sentiment (scale -1 to 1): {news_data['average_sentiment']:.2f}
        
        Recent News Headlines:
        {simplified_news}
        
        Provide a 2-3 sentence investment recommendation.
        """
        
        # Check if Ollama is available, use a fallback if not
        try:
            response = ollama.generate(model="qwen2.5:0.5b", prompt=query)
            return response['response']
        except Exception as ollama_error:
            logger.warning(f"Ollama error: {ollama_error}. Using fallback response.")
            # Fallback response
            sentiment_word = "positive" if news_data['average_sentiment'] > 0 else "negative" if news_data['average_sentiment'] < 0 else "neutral"
            return f"Based on {sentiment_word} news sentiment and technical analysis suggesting a {market_trend['trend']} signal, consider following the {market_trend['trend']} recommendation. Always conduct additional research before making investment decisions."
            
    except Exception as e:
        logger.error(f"Failed to generate insights: {str(e)}")
        return "Unable to generate insights due to an error. Please rely on the trend analysis and conduct your own research."

# Step 4: Execute Trade (simulation only)
def execute_trade(decision: str, symbol: str) -> Dict[str, str]:
    """
    Simulate trade execution.
    """
    logger.info(f"Executing trade: {decision} {symbol}")
    # In a real application, you would add API call to execute trade here
    return {"action": decision, "symbol": symbol, "status": "simulated"}

# Step 5: FastAPI Endpoint
@app.get("/trade/{symbol}")
def trade(symbol: str):
    try:
        logger.info(f"Fetching market data for symbol: {symbol}")

        # Fetch market data using yfinance
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")  # Adjust the period as needed

        if df.empty:
            raise ValueError(f"No historical data found for the symbol '{symbol}'.")

        # Gather news and perform sentiment analysis
        news_data = gather_news_and_sentiment(symbol)

        # Analyze market trends
        market_trends = analyze_market_trends(df)
        market_trends["symbol"] = symbol  # Add symbol to market trends for Ollama

        # Generate insights
        insights = generate_insights_with_ollama(news_data, market_trends)

        # Execute trade based on trend (simulation)
        trade_result = None
        if market_trends['trend'] == 'buy':
            trade_result = execute_trade("Buy", symbol)
        elif market_trends['trend'] == 'sell':
            trade_result = execute_trade("Sell", symbol)
        else:
            logger.info(f"Hold recommendation for {symbol}")
            trade_result = {"action": "Hold", "symbol": symbol, "status": "no action taken"}

        return {
            "status": "success",
            "symbol": symbol,
            "trend": market_trends['trend'],
            "last_price": market_trends['last_price'],
            "predicted_price": market_trends['predicted_price'],
            "news_sentiment": news_data['average_sentiment'],
            "insights": insights,
            "trade_action": trade_result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run the app, use: `uvicorn filename:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)