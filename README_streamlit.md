# Technical Indicator Trading Strategy - Streamlit Web Application

## Overview

This is a comprehensive Streamlit web application that allows users to analyze stock trading strategies using technical indicators and machine learning. The application combines the functionality of both model1.py and model2.py with an interactive web interface.

## Features

### 🎯 Core Functionality

- **Ticker Input**: Enter any stock ticker symbol (e.g., SPY, AAPL, MSFT, TSLA)
- **Flexible Date Ranges**: Choose between default 10-year period or custom date ranges
- **Model Selection**: Choose between Random Forest and Gradient Boosting models
- **Three-Day Confirmation**: Optional signal confirmation to reduce false signals
- **Interactive Visualizations**: Real-time charts with zoom, pan, and hover capabilities
- **Performance Metrics**: Comprehensive dashboard with key trading metrics
- **Data Export**: Download results as CSV for further analysis

### 📊 Technical Indicators

- **RSI** (Relative Strength Index) - 14-day window
- **MACD** (Moving Average Convergence Divergence) - 12/26/9 parameters
- **Bollinger Bands** - 20-day window, 2 standard deviations
- **Simple Moving Averages** - 5, 10, 20, 50, 100, 200-day periods
- **Momentum Indicators** - 10 and 20-day price changes

### 📈 Performance Metrics

- **CAGR** (Compound Annual Growth Rate)
- **Volatility** (Annualized standard deviation)
- **Sharpe Ratio** (Risk-adjusted returns)
- **Maximum Drawdown** (Peak-to-trough decline)
- **Hit Rate** (Percentage of profitable trades)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the application files**
2. **Install required packages**:

   ```bash
   pip install streamlit pandas numpy yfinance scikit-learn plotly matplotlib
   ```

3. **Run the application**:

   ```bash
   streamlit run streamlit_trading_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Requirements File

Create a `requirements.txt` file with the following content:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
yfinance>=0.2.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.7.0
```

## Usage Guide

### 1. Basic Usage

1. Enter a stock ticker symbol in the sidebar
2. Configure strategy parameters (or use defaults)
3. Click "Run Strategy" to execute
4. View results in the main display area

### 2. Advanced Configuration

#### Date Range Settings

- **Default Mode**: Uses 10 years of historical data
- **Custom Mode**: Select specific start/end dates
- **Training Period**: Set the cutoff date for train/test split

#### Model Configuration

- **Random Forest (RF)**: Ensemble method with multiple decision trees
- **Gradient Boosting (GB)**: Sequential tree building with error correction

#### Strategy Parameters

- **Probability Threshold**: Minimum confidence level for signals (0.5-0.8)
- **Three-Day Confirmation**: Requires 3 consecutive days of same signal
- **Transaction Fee**: Trading costs in basis points (0-10 bps)

### 3. Interpreting Results

#### Model Performance

- **CV AUC**: Cross-validated area under curve (higher is better)
- **Best Parameters**: Optimized hyperparameters from grid search

#### Strategy Performance

- **Green metrics**: Positive performance indicators
- **Red metrics**: Areas needing improvement
- **Interactive charts**: Click and drag to explore specific time periods

## Technical Architecture

### Data Flow

1. **Data Fetching**: Retrieve stock data from Yahoo Finance API
2. **Feature Engineering**: Calculate technical indicators
3. **Model Training**: Train ML model on historical data
4. **Signal Generation**: Predict future price movements
5. **Backtesting**: Simulate trading strategy with transaction costs
6. **Performance Analysis**: Calculate key metrics and visualizations

### Machine Learning Pipeline

```python
# Feature Engineering
Technical Indicators → Feature Matrix → Target Labels

# Model Training
Grid Search CV → Hyperparameter Tuning → Best Model

# Strategy Execution
Predictions → Signal Confirmation → Trading Simulation
```

### Key Functions

- `fetch_prices()`: Yahoo Finance data retrieval
- `make_features()`: Technical indicator calculation
- `make_model()`: ML model initialization
- `backtest()`: Strategy simulation with costs
- `calculate_performance()`: Metrics computation

## Deployment Options

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_trading_app.py
```

### Cloud Deployment

#### Streamlit Cloud (Recommended)

1. Upload code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

#### Heroku

```bash
# Create Procfile
web: streamlit run streamlit_trading_app.py --server.port=$PORT

# Deploy
heroku create your-app-name
git push heroku main
```

#### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_trading_app.py", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

#### Data Fetching Errors

- **Problem**: No data found for ticker
- **Solution**: Verify ticker symbol and check market availability
- **Alternative**: Try different date ranges

#### Model Training Issues

- **Problem**: Insufficient training data
- **Solution**: Extend date range or reduce training period
- **Alternative**: Use simpler model parameters

#### Performance Issues

- **Problem**: Slow execution
- **Solution**: Reduce date range or use fewer hyperparameters
- **Alternative**: Enable caching in Streamlit

### Error Messages

- **"No data found for ticker"**: Check ticker symbol validity
- **"Insufficient data for training"**: Adjust date range settings
- **"Model training failed"**: Try default parameters

## Advanced Features

### Custom Indicators

Users can extend the `make_features()` function to include:

- Custom moving averages
- Volume-based indicators
- Volatility measures
- Pattern recognition signals

### Portfolio Analysis

Future enhancements may include:

- Multi-asset portfolio optimization
- Correlation analysis
- Risk management tools
- Rebalancing strategies

### Real-time Integration

Potential additions:

- Live price feeds
- Real-time signal generation
- Automated trading integration
- Alert systems

## Security Considerations

### Data Privacy

- No user data storage
- Secure API connections
- Rate limiting implemented

### Input Validation

- Ticker symbol verification
- Date range validation
- Parameter bounds checking

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

### Code Standards

- Follow PEP 8 guidelines
- Add comprehensive comments
- Include error handling
- Test edge cases

## License

This project is open source and available under the MIT License.

## Support

For questions or issues:

1. Check troubleshooting section
2. Review documentation
3. Create GitHub issue
4. Contact development team

## Version History

### v1.0.0 (Current)

- Basic ticker input and analysis
- Random Forest and Gradient Boosting models
- Three-day confirmation feature
- Interactive visualizations
- Performance metrics dashboard
- CSV export functionality

### Future Roadmap

- Multi-asset portfolio support
- Real-time data integration
- Advanced technical indicators
- Strategy comparison tools
- Mobile optimization

---

**Disclaimer**: This application is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough research before making investment decisions.
