import streamlit as st
import plotly.graph_objects as go
from datagen.financial import TickGenerator, MarketRegime, AssetClass, MarketHours
import pandas as pd
from datetime import datetime
import os

# Set page config for dark theme
st.set_page_config(
    page_title="Market Data Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme to all Plotly figures
plotly_layout = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stSelectbox, .stNumberInput {
            background-color: #262730;
        }
        .stButton>button {
            background-color: #262730;
            color: #FAFAFA;
        }
    </style>
""", unsafe_allow_html=True)

def generate_data(params):
    """Generate market data with given parameters."""
    # Separate generator params from generation params
    generator_params = {
        'regime': params['regime'],
        'asset_class': params['asset_class'],
        'market_hours': params['market_hours'],
        'base_volatility': params['base_volatility'],
        'base_drift': params['base_drift'],
        'mean_reversion_strength': params['mean_reversion'],
        'momentum_factor': params['momentum'],
        'jump_intensity': params['jump_intensity'],
        'jump_size_mean': params['jump_size_mean'],
        'jump_size_std': params['jump_size_std'],
        'volatility_of_volatility': params['vol_of_vol'],
        'ticks_per_day': 390000,
    }
    
    gen = TickGenerator(**generator_params)
    gen.generate(
        days=params['days'],
        start_date=params['start_date'],
        start_price=params['start_price']
    )
    
    # Save data at different timeframes
    data = {}
    for freq in ["1min", "5min", "15min", "30min", "1H", "1D"]:
        filename = f"market_data_{freq}"
        gen.save(filename, freq=freq)
        data[freq] = pd.read_csv(f"output/{filename}.csv")
        data[freq]['datetime'] = pd.to_datetime(data[freq]['datetime'])
    
    return data

def plot_ohlc(data, title="OHLC Candlestick Chart"):
    """Create an OHLC candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=data['datetime'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Time',
        **plotly_layout,
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='category'
        ),
        height=600
    )
    
    return fig

def main():
    st.title("📈 Market Data Generator")
    
    # Sidebar for parameters
    st.sidebar.header("Generator Parameters")
    
    # Market regime selection
    regime = st.sidebar.selectbox(
        "Market Regime",
        options=[r.value for r in MarketRegime],
        format_func=lambda x: x.title()
    )
    
    # Asset class selection
    asset_class = st.sidebar.selectbox(
        "Asset Class",
        options=[a.value for a in AssetClass],
        format_func=lambda x: x.title()
    )
    
    # Market hours selection
    market_hours = st.sidebar.selectbox(
        "Market Hours",
        options=[m.value for m in MarketHours],
        format_func=lambda x: x.title()
    )
    
    # Basic parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        base_volatility = st.number_input("Base Volatility", value=0.02, format="%.3f", step=0.001)
        base_drift = st.number_input("Base Drift", value=-0.0001, format="%.4f", step=0.0001)
        mean_reversion = st.number_input("Mean Reversion", value=0.002, format="%.3f", step=0.001)
        momentum = st.number_input("Momentum", value=0.02, format="%.3f", step=0.001)
    
    with col2:
        jump_intensity = st.number_input("Jump Intensity", value=0.0005, format="%.4f", step=0.0001)
        jump_size_mean = st.number_input("Jump Size Mean", value=-0.005, format="%.3f", step=0.001)
        jump_size_std = st.number_input("Jump Size Std", value=0.002, format="%.3f", step=0.001)
        vol_of_vol = st.number_input("Vol of Vol", value=0.2, format="%.2f", step=0.01)
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    days = st.sidebar.number_input("Number of Days", value=10, min_value=1, max_value=100)
    start_price = st.sidebar.number_input("Start Price", value=100.0, min_value=1.0)
    start_date = st.sidebar.date_input("Start Date", datetime.now())
    
    # Generate button
    if st.sidebar.button("Generate Data"):
        # Show loading spinner
        with st.spinner("Generating market data..."):
            params = {
                'regime': MarketRegime(regime),
                'asset_class': AssetClass(asset_class),
                'market_hours': MarketHours(market_hours),
                'base_volatility': base_volatility,
                'base_drift': base_drift,
                'mean_reversion': mean_reversion,
                'momentum': momentum,
                'jump_intensity': jump_intensity,
                'jump_size_mean': jump_size_mean,
                'jump_size_std': jump_size_std,
                'vol_of_vol': vol_of_vol,
                'days': days,
                'start_date': start_date,
                'start_price': start_price
            }
            
            # Generate and store data in session state
            st.session_state.data = generate_data(params)
            st.session_state.current_timeframe = "1H"  # Default timeframe
    
    # Main area for chart
    st.subheader("Market Data Visualization")
    
    # Timeframe selection
    if 'data' in st.session_state:
        timeframe = st.selectbox(
            "Select Timeframe",
            options=["1min", "5min", "15min", "30min", "1H", "1D"],
            index=["1min", "5min", "15min", "30min", "1H", "1D"].index(st.session_state.current_timeframe)
        )
        st.session_state.current_timeframe = timeframe
        
        # Display chart
        fig = plot_ohlc(
            st.session_state.data[timeframe],
            f"OHLC Candlestick Chart ({timeframe} timeframe)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data statistics
        data = st.session_state.data[timeframe]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
        with col2:
            st.metric("High", f"${data['high'].max():.2f}")
        with col3:
            st.metric("Low", f"${data['low'].min():.2f}")
        with col4:
            st.metric("Volatility", f"{data['close'].pct_change().std() * 100:.2f}%")
    else:
        st.info("Click 'Generate Data' to create and visualize market data.")

if __name__ == "__main__":
    main()
