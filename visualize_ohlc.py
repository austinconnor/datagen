import pandas as pd
import plotly.graph_objects as go
import argparse

def visualize_ohlc(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'])])
    
    # Update the layout
    fig.update_layout(
        title='OHLC Candlestick Chart',
        yaxis_title='Price',
        xaxis_title='Time',
        template='plotly_dark',
        xaxis=dict(
            rangeslider=dict(visible=False),  # Disable rangeslider to avoid gaps
            type='category'  # Use category type to show data continuously
        )
    )
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize OHLC data from a CSV file')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing OHLC data')
    args = parser.parse_args()
    
    visualize_ohlc(args.csv_file)
