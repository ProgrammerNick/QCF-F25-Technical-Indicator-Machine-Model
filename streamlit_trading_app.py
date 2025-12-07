import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Technical Indicator Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success {
        color: #00a86b;
    }
    .danger {
        color: #dc3912;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Config:
    ticker: str = "SPY"
    period: str = "10y"
    start: str = None
    end: str = None
    train_end: str = "2024-12-31"
    proba_threshold: float = 0.55
    fee_bps: float = 1.0
    model: str = "rf"
    random_state: int = 42
    use_confirmation: bool = True

def fetch_prices(ticker, start=None, end=None, period="10y"):
    """Fetch price data from Yahoo Finance"""
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        elif end and not start:
            end_dt = pd.to_datetime(end)
            start_dt = end_dt - pd.DateOffset(years=10)
            df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end, auto_adjust=True, progress=False)
        elif start and not end:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        else:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        
        if df.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
            
        # Flatten MultiIndex columns and rename
        df.columns = [col[0].lower() for col in df.columns]
        df["ret1"] = df["close"].pct_change()
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def rsi(close, window=14):
    """Calculate RSI indicator"""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def bollinger(close, n=20, k=2.0):
    """Calculate Bollinger Bands"""
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k*sd
    low = ma - k*sd
    width = (up - low) / (ma + 1e-12)
    pctb = (close - low) / ((up - low) + 1e-12)
    return ma, up, low, width, pctb

def make_features(df):
    """Create technical indicator features"""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    out["rsi"] = rsi(c, 14)
    ml, ms, mh = macd(c)
    out["macd_line"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    ma, up, low, width, pctb = bollinger(c, 20, 2.0)
    out["bb_pctb"], out["bb_width"], out["bb_ma"] = pctb, width, ma
    for w in [5,10,20,50,100,200]:
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"slope_{w}"] = out[f"sma_{w}"].pct_change(5)
    out["mom_10"], out["mom_20"] = c.pct_change(10), c.pct_change(20)
    return out.dropna()

def make_labels(df, horizon=1):
    """Create binary labels for classification"""
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)

def make_model(name, random_state=42):
    """Create machine learning model with hyperparameters"""
    if name == "rf":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8],
            "model__min_samples_leaf": [5, 10],
        }
    elif name == "gb":
        model = GradientBoostingClassifier(random_state=random_state)
        grid = {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    else:
        raise ValueError("unknown model")
    pipe = Pipeline([("model", model)])
    return pipe, grid

def backtest(close, signal, fee_bps=1.0):
    """Perform backtesting of trading strategy"""
    close = close.loc[signal.index]
    ret1 = close.pct_change().fillna(0.0)
    sig = signal.reindex(close.index).fillna(0.0).astype(float)
    turnover = sig.diff().abs().fillna(0.0)
    fee = turnover * (fee_bps/10000.0)
    strat_ret = sig.shift(1).fillna(0.0) * ret1 - fee
    eq = (1 + strat_ret).cumprod()
    bh = (1 + ret1).cumprod()
    return strat_ret, eq, bh

def calculate_performance(strat_ret):
    """Calculate performance metrics"""
    ann = 252
    if len(strat_ret) == 0:
        return {k: np.nan for k in ["CAGR","Vol","Sharpe","MaxDD","HitRate"]}
    eq = (1 + strat_ret).cumprod()
    cagr = eq.iloc[-1]**(ann/len(eq)) - 1
    vol = strat_ret.std() * np.sqrt(ann)
    sharpe = (strat_ret.mean()/(strat_ret.std()+1e-12)) * np.sqrt(ann)
    mdd = (eq/eq.cummax() - 1).min()
    hit = (strat_ret > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd, "HitRate": hit}

def create_equity_curve_chart(eq, bh, ticker, prices):
    """Create interactive equity curve chart comparing strategy vs buy & hold"""
    # Handle both Series and DataFrame for prices
    if isinstance(prices, pd.Series):
        price_values = prices.values
        price_index = prices.index
    else:
        price_values = prices['close'].values
        price_index = prices.index
    
    # Create buy & hold returns from price data
    buy_hold_returns = price_values / price_values[0]
    
    # Create figure
    fig = go.Figure()
    
    # Add buy & hold line
    fig.add_trace(go.Scatter(
        x=price_index,
        y=buy_hold_returns,
        name='Buy & Hold',
        line=dict(color='red', dash='dash')
    ))
    
    # Add strategy equity curve
    fig.add_trace(go.Scatter(
        x=eq.index,
        y=eq.values,
        name='Strategy',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title=f'{ticker} - Strategy vs Buy & Hold Performance',
        xaxis_title='Date',
        yaxis_title='Normalized Returns (Starting at 1.0)',
        hovermode='x unified',
        legend=dict(x=0, y=1),
        height=600
    )
    return fig

def create_signals_chart(prices, buy_points, sell_points, ticker):
    """Create interactive signals chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices['close'], name=f'{ticker} Close Price', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['close'], mode='markers', name='Buy Signal', 
                           marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['close'], mode='markers', name='Sell Signal', 
                           marker=dict(symbol='triangle-down', size=10, color='red')))
    fig.update_layout(
        title=f'{ticker} Price with Buy/Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified'
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">📈 Technical Indicator Trading Strategy</h1>', unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.header("Strategy Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY", help="Enter stock ticker symbol (e.g., SPY, AAPL, MSFT)")
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range")
    
    if use_custom_dates:
        start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=3650))
        end_date = st.sidebar.date_input("End Date", value=datetime.now())
        train_end_date = st.sidebar.date_input("Training End Date", value=datetime.now() - timedelta(days=365))
    else:
        start_date = None
        end_date = None
        train_end_date = None
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    model_type = st.sidebar.selectbox("Select Model Type", ["rf", "gb"], 
                                    help="RF: Random Forest, GB: Gradient Boosting")
    
    # Three-day confirmation toggle
    use_confirmation = st.sidebar.checkbox("Use 3-Day Confirmation", value=True,
                                        help="Require 3 consecutive days of same signal before executing trade")
    
    # Probability threshold
    proba_threshold = st.sidebar.slider("Probability Threshold", 0.5, 0.8, 0.55, 0.05,
                                       help="Minimum probability threshold for generating signals")
    
    # Fee configuration
    fee_bps = st.sidebar.slider("Transaction Fee (bps)", 0, 10, 1, 1,
                               help="Transaction fee in basis points")
    
    # Run button
    run_strategy = st.sidebar.button("🚀 Run Strategy", type="primary")
    
    if run_strategy:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update configuration
            config = Config(
                ticker=ticker.upper(),
                start=start_date.strftime('%Y-%m-%d') if start_date else None,
                end=end_date.strftime('%Y-%m-%d') if end_date else None,
                train_end=train_end_date.strftime('%Y-%m-%d') if train_end_date else "2024-12-31",
                proba_threshold=proba_threshold,
                fee_bps=fee_bps,
                model=model_type,
                random_state=42,
                use_confirmation=use_confirmation
            )
            
            status_text.text("Fetching price data...")
            progress_bar.progress(10)
            
            # Fetch data
            px = fetch_prices(config.ticker, config.start, config.end, config.period)
            if px is None:
                return
            
            status_text.text("Creating technical indicators...")
            progress_bar.progress(30)
            
            # Create features and labels
            X_all = make_features(px)
            y_all = make_labels(px)
            idx = X_all.index.intersection(y_all.index)
            X_all = X_all.loc[idx].shift(1).dropna()
            y_all = y_all.loc[X_all.index]
            
            status_text.text("Splitting data...")
            progress_bar.progress(40)
            
            # Split data
            train_mask = X_all.index <= config.train_end
            X_train, X_test = X_all[train_mask], X_all[~train_mask]
            y_train, y_test = y_all[train_mask], y_all[~train_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("Insufficient data for training or testing. Please adjust date range.")
                return
            
            status_text.text("Training model...")
            progress_bar.progress(60)
            
            # Create and train model
            pipe, grid = make_model(config.model, config.random_state)
            tscv = TimeSeriesSplit(n_splits=5)
            gcv = GridSearchCV(pipe, grid, cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=0)
            gcv.fit(X_train, y_train)
            
            best = gcv.best_estimator_
            best.fit(X_train, y_train)
            
            status_text.text("Generating signals...")
            progress_bar.progress(80)
            
            # Generate signals
            if len(X_test) > 0:
                proba = best.predict_proba(X_test)
                pos_col = list(best.named_steps["model"].classes_).index(1)
                p_up = pd.Series(proba[:, pos_col], index=X_test.index)
                raw_signal = (p_up >= config.proba_threshold).astype(int)
                
                if config.use_confirmation:
                    signal = (raw_signal.rolling(window=3).sum() == 3).astype(int)
                else:
                    signal = raw_signal
            else:
                p_up = pd.Series(dtype=float)
                signal = pd.Series(dtype=int)
            
            status_text.text("Backtesting strategy...")
            progress_bar.progress(90)
            
            # Backtest
            strat_ret, eq, bh = backtest(px.loc[signal.index, "close"], signal, fee_bps=config.fee_bps)
            metrics = calculate_performance(strat_ret)
            
            status_text.text("Complete!")
            progress_bar.progress(100)
            
            # Display results
            st.success("Strategy execution completed successfully!")
            
            # Model performance
            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best CV AUC", f"{gcv.best_score_:.4f}")
            with col2:
                st.metric("Training Samples", len(X_train))
                st.metric("Test Samples", len(X_test))
            
            # Best parameters in an expandable section
            with st.expander("🔧 Best Model Parameters"):
                st.json(gcv.best_params_)
            
            # Strategy performance metrics
            st.subheader("💰 Strategy Performance")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                cagr_color = "success" if metrics["CAGR"] > 0 else "danger"
                st.markdown(f'<div class="metric-card"><strong>CAGR</strong><br><span class="{cagr_color}">{metrics["CAGR"]:.2%}</span></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card"><strong>Volatility</strong><br>{metrics["Vol"]:.2%}</div>', unsafe_allow_html=True)
            
            with col3:
                sharpe_color = "success" if metrics["Sharpe"] > 1 else "danger"
                st.markdown(f'<div class="metric-card"><strong>Sharpe Ratio</strong><br><span class="{sharpe_color}">{metrics["Sharpe"]:.2f}</span></div>', unsafe_allow_html=True)
            
            with col4:
                dd_color = "danger" if metrics["MaxDD"] < -0.1 else "success"
                st.markdown(f'<div class="metric-card"><strong>Max Drawdown</strong><br><span class="{dd_color}">{metrics["MaxDD"]:.2%}</span></div>', unsafe_allow_html=True)
            
            with col5:
                hit_color = "success" if metrics["HitRate"] > 0.5 else "danger"
                st.markdown(f'<div class="metric-card"><strong>Hit Rate</strong><br><span class="{hit_color}">{metrics["HitRate"]:.2%}</span></div>', unsafe_allow_html=True)
            
            # Performance comparison visualization
            st.subheader("📈 Performance Comparison")
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                # Calculate market metrics from buy & hold
                market_ret = px.loc[signal.index, "ret1"]
                market_eq = (1 + market_ret).cumprod()
                market_metrics = calculate_performance(market_ret)
                
                # Create bar chart for comparison with proper legend
                fig_metrics = go.Figure()
                
                # Add strategy bars for all metrics
                fig_metrics.add_trace(go.Bar(
                    name='Strategy',
                    x=['CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Hit Rate'],
                    y=[metrics["CAGR"], metrics["Vol"], metrics["Sharpe"], abs(metrics["MaxDD"]), metrics["HitRate"]],
                    marker_color='blue',
                    offsetgroup=0
                ))
                
                # Add market bars for all metrics
                fig_metrics.add_trace(go.Bar(
                    name='Market (Buy & Hold)',
                    x=['CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Hit Rate'],
                    y=[market_metrics["CAGR"], market_metrics["Vol"], market_metrics["Sharpe"], abs(market_metrics["MaxDD"]), (market_ret > 0).mean()],
                    marker_color='red',
                    offsetgroup=1
                ))
                
                fig_metrics.update_layout(
                    title='Strategy vs Market Performance',
                    xaxis_title='Metrics',
                    yaxis_title='Value',
                    barmode='group',
                    height=400,
                    legend=dict(
                        title="Legend",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            with perf_col2:
                # Display detailed metrics in a clean format
                st.markdown("### Detailed Performance Metrics")
                
                # Create a comparison table
                comparison_data = {
                    'Metric': ['CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Hit Rate'],
                    'Strategy': [
                        f"{metrics['CAGR']:.2%}",
                        f"{metrics['Vol']:.2%}",
                        f"{metrics['Sharpe']:.2f}",
                        f"{metrics['MaxDD']:.2%}",
                        f"{metrics['HitRate']:.2%}"
                    ],
                    'Market (Buy & Hold)': [
                        f"{market_metrics['CAGR']:.2%}",
                        f"{market_metrics['Vol']:.2%}",
                        f"{market_metrics['Sharpe']:.2f}",
                        f"{market_metrics['MaxDD']:.2%}",
                        f"{(market_ret > 0).mean():.2%}"
                    ],
                    'Outperformance': [
                        f"{(metrics['CAGR'] - market_metrics['CAGR']):.2%}",
                        f"{(metrics['Vol'] - market_metrics['Vol']):.2%}",
                        f"{(metrics['Sharpe'] - market_metrics['Sharpe']):.2f}",
                        f"{(metrics['MaxDD'] - market_metrics['MaxDD']):.2%}",
                        f"{(metrics['HitRate'] - (market_ret > 0).mean()):.2%}"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Add performance summary
                st.markdown("### Performance Summary")
                if metrics["CAGR"] > market_metrics["CAGR"]:
                    st.success(f"✅ Strategy outperforms market by {metrics['CAGR'] - market_metrics['CAGR']:.2%} CAGR")
                else:
                    st.error(f"❌ Strategy underperforms market by {market_metrics['CAGR'] - metrics['CAGR']:.2%} CAGR")
                
                if metrics["Sharpe"] > market_metrics["Sharpe"]:
                    st.success(f"✅ Better risk-adjusted returns (Sharpe: {metrics['Sharpe']:.2f} vs {market_metrics['Sharpe']:.2f})")
                else:
                    st.error(f"❌ Lower risk-adjusted returns (Sharpe: {metrics['Sharpe']:.2f} vs {market_metrics['Sharpe']:.2f})")
            
            # Charts
            st.subheader("📈 Visualizations")
            
            # Equity curve with price
            st.plotly_chart(create_equity_curve_chart(eq, bh, config.ticker, px.loc[signal.index, 'close']), use_container_width=True)
            
            # Signals chart
            if len(signal) > 0:
                prices = px.loc[signal.index]
                shifted_signal = signal.shift(1).fillna(0)
                buy_signals = (signal == 1) & (shifted_signal == 0)
                sell_signals = (signal == 0) & (shifted_signal == 1)
                buy_points = prices[buy_signals]
                sell_points = prices[sell_signals]
                
                st.plotly_chart(create_signals_chart(prices, buy_points, sell_points, config.ticker), use_container_width=True)
            
            # Data table
            st.subheader("📋 Signal Data")
            out = pd.DataFrame({
                "Date": signal.index,
                "Probability_Up": p_up,
                "Signal": signal,
                "Return": px.loc[signal.index, "ret1"],
                "Strategy_Return": strat_ret
            })
            st.dataframe(out, use_container_width=True)
            
            # Download button
            csv = out.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{config.ticker}_strategy_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
