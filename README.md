# 📈 Technical Indicator Trading Strategy - Streamlit Web Application

A comprehensive web application for analyzing stock trading strategies using technical indicators and machine learning. This application combines the functionality of both model1.py and model2.py with an interactive, user-friendly interface.

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)

```bash
# Clone or download the files
# Run the setup script
python run_app.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_trading_app.py
```

The application will open automatically in your web browser at `http://localhost:8501`

## ✨ Features

### 🎯 Core Functionality

- **Ticker Input**: Enter any stock ticker symbol (SPY, AAPL, MSFT, TSLA, etc.)
- **Flexible Date Ranges**: Default 10-year data or custom date selection
- **Model Selection**: Choose between Random Forest and Gradient Boosting
- **Three-Day Confirmation**: Optional signal confirmation to reduce false signals
- **Interactive Charts**: Real-time visualizations with zoom and pan
- **Performance Metrics**: Comprehensive trading strategy dashboard
- **Data Export**: Download results as CSV

### 📊 Technical Indicators

- **RSI** (Relative Strength Index) - 14-day window
- **MACD** (Moving Average Convergence Divergence) - 12/26/9 parameters
- **Bollinger Bands** - 20-day window, 2 standard deviations
- **Simple Moving Averages** - 5, 10, 20, 50, 100, 200-day periods
- **Momentum Indicators** - 10 and 20-day price changes

### 💹 Performance Metrics

- **CAGR** (Compound Annual Growth Rate)
- **Volatility** (Annualized standard deviation)
- **Sharpe Ratio** (Risk-adjusted returns)
- **Maximum Drawdown** (Peak-to-trough decline)
- **Hit Rate** (Percentage of profitable trades)

## 🖥️ Application Interface

### Sidebar Configuration

- **Ticker Input**: Stock symbol entry with validation
- **Date Range**: Custom date selection or default 10-year period
- **Model Settings**: RF/GB selection, probability threshold, confirmation toggle
- **Transaction Costs**: Fee configuration in basis points

### Main Display

- **Progress Tracking**: Real-time execution progress
- **Model Performance**: CV AUC, best parameters, sample sizes
- **Strategy Dashboard**: Color-coded performance metrics
- **Interactive Visualizations**: Equity curves and signal charts
- **Data Table**: Complete signal history with export option

## 📁 File Structure

```
├── streamlit_trading_app.py    # Main application file
├── requirements.txt            # Python dependencies
├── run_app.py               # Easy setup script
├── README.md                 # This file
├── model1.py                # Original model 1
├── model2.py                # Original model 2
├── TEST.py                  # Test model with 3-day confirmation
└── streamlit_app_plan.md      # Detailed technical documentation
```

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Data Fetching**: Yahoo Finance API integration
2. **Feature Engineering**: Technical indicator calculation
3. **Model Training**: Grid search with time series cross-validation
4. **Signal Generation**: Probability-based predictions with confirmation
5. **Backtesting**: Strategy simulation with transaction costs
6. **Performance Analysis**: Comprehensive metrics calculation

### Three-Day Confirmation Logic

```python
# Raw signal generation
raw_signal = (p_up >= proba_threshold).astype(int)

# Three-day confirmation
signal = (raw_signal.rolling(window=3).sum() == 3).astype(int)
```

This feature reduces false signals by requiring three consecutive days of the same prediction before executing trades.

## 📊 Usage Examples

### Basic Analysis

1. Enter "SPY" as ticker
2. Use default settings
3. Click "Run Strategy"
4. View results and performance metrics

### Advanced Configuration

1. Enter custom ticker (e.g., "AAPL")
2. Enable "Use Custom Date Range"
3. Select specific date period
4. Choose Gradient Boosting model
5. Enable 3-day confirmation
6. Set probability threshold to 0.6
7. Click "Run Strategy"

## 🌐 Deployment Options

### Local Development

```bash
# Using setup script
python run_app.py

# Manual execution
streamlit run streamlit_trading_app.py
```

### Cloud Deployment

#### Streamlit Cloud (Easiest)

1. Upload code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

#### Heroku

```bash
# Create Procfile
echo "web: streamlit run streamlit_trading_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-trading-app
git push heroku main
```

#### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_trading_app.py", "--server.address=0.0.0.0"]
```

## 🛠️ Customization

### Adding New Indicators

```python
def custom_indicator(close, window=20):
    # Your custom logic here
    return indicator_values

# Add to make_features function
out["custom"] = custom_indicator(c)
```

### Modifying Model Parameters

```python
# Update make_model function grid
grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__min_samples_leaf": [1, 3, 5],
}
```

## 🔍 Troubleshooting

### Common Issues

#### Data Fetching Errors

- **Problem**: "No data found for ticker"
- **Solution**: Verify ticker symbol and market availability
- **Alternative**: Try different date ranges

#### Performance Issues

- **Problem**: Slow execution
- **Solution**: Reduce date range or use fewer hyperparameters
- **Alternative**: Enable caching in Streamlit

#### Memory Issues

- **Problem**: Application crashes with large datasets
- **Solution**: Limit date range to 5 years or less
- **Alternative**: Use cloud deployment with more resources

### Error Messages

- **"Insufficient data for training"**: Extend date range or reduce training period
- **"Model training failed"**: Use default parameters or simpler model
- **"Connection error"**: Check internet connection and Yahoo Finance status

## 📈 Performance Optimization

### Recommended Settings

- **Date Range**: 5-10 years for optimal balance
- **Probability Threshold**: 0.55-0.65 for good signal quality
- **Confirmation**: Enable for volatile stocks, disable for trending stocks
- **Transaction Fees**: 1-5 bps for realistic simulation

### Model Selection

- **Random Forest**: Better for noisy data, more stable
- **Gradient Boosting**: Better for complex patterns, higher accuracy

## 🔒 Security & Privacy

- No user data storage
- Secure API connections
- Input validation and sanitization
- Rate limiting implemented
- No external data sharing

## 📚 Dependencies

### Core Libraries

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **yfinance**: Financial data fetching
- **scikit-learn**: Machine learning
- **plotly**: Interactive visualizations
- **matplotlib**: Additional plotting capabilities

### Version Requirements

See [`requirements.txt`](requirements.txt) for specific version requirements.

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Make changes with proper testing
4. Submit pull request with description

### Code Standards

- Follow PEP 8 guidelines
- Add comprehensive comments
- Include error handling
- Test edge cases
- Update documentation

## 📄 License

This project is open source under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

**Educational Purpose Only**: This application is for research and educational purposes. Past performance does not guarantee future results. Always conduct thorough research before making investment decisions.

**Financial Risk**: Trading involves substantial risk of loss. Use this application at your own risk.

**Data Accuracy**: While we strive for accuracy, financial data may contain errors or delays.

## 📞 Support

For questions, issues, or contributions:

1. Check this README and troubleshooting section
2. Review technical documentation in `streamlit_app_plan.md`
3. Create GitHub issue with detailed description
4. Contact development team for urgent matters

## 🔄 Version History

### v1.0.0 (Current)

- ✅ Complete Streamlit web application
- ✅ All features from model1.py and model2.py
- ✅ Three-day confirmation implementation
- ✅ Interactive visualizations with Plotly
- ✅ Comprehensive performance metrics
- ✅ CSV export functionality
- ✅ Easy setup script
- ✅ Professional UI/UX design

### Future Roadmap

- 🔄 Multi-asset portfolio support
- 🔄 Real-time data integration
- 🔄 Advanced technical indicators
- 🔄 Strategy comparison tools
- 🔄 Mobile optimization
- 🔄 Backtesting enhancements

---

**Made with ❤️ for quantitative finance and machine learning enthusiasts**

If you find this project useful, please consider giving it a ⭐ on GitHub!
