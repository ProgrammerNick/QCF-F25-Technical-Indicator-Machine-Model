
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def plot_signals(csv_path="test_set_predictions_signals.csv", ticker="SPY", output_image="spy_signals_plot.png"):
    """
    Generates a plot of the ticker's price with buy and sell signals overlaid.

    Args:
        csv_path (str): Path to the CSV file with signals.
        ticker (str): The stock ticker symbol.
        output_image (str): The filename for the output plot image.
    """
    # Load the signals
    signals_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

    if signals_df.empty:
        print("Signal file is empty. No plot to generate.")
        return

    # Fetch the price data for the same period
    start_date = signals_df.index.min().strftime('%Y-%m-%d')
    end_date = signals_df.index.max().strftime('%Y-%m-%d')
    prices = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if prices.empty:
        print(f"Could not download price data for {ticker}. Cannot generate plot.")
        return

    # Align signals with the price data
    signals_df = signals_df.reindex(prices.index)

    # Identify buy and sell points
    signals_df['signal'] = signals_df['signal'].fillna(0) # Forward fill to handle non-trading days
    signals_df['shifted_signal'] = signals_df['signal'].shift(1).fillna(0)

    buy_signals = (signals_df['signal'] == 1) & (signals_df['shifted_signal'] == 0)
    sell_signals = (signals_df['signal'] == 0) & (signals_df['shifted_signal'] == 1)

    buy_points = prices[buy_signals]
    sell_points = prices[sell_signals]

    # Create the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot the close price
    ax.plot(prices.index, prices['Close'], label=f'{ticker} Close Price', color='skyblue', linewidth=2)

    # Plot buy signals
    ax.plot(buy_points.index, buy_points['Close'], '^', markersize=10, color='green', label='Buy Signal', alpha=0.8)

    # Plot sell signals
    ax.plot(sell_points.index, sell_points['Close'], 'v', markersize=10, color='red', label='Sell Signal', alpha=0.8)

    # Formatting
    ax.set_title(f'{ticker} Price with Buy/Sell Signals', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    plot_signals()
