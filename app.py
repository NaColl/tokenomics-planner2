import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Tokenomics Planner", layout="wide")

# Initialize session state
if 'distribution' not in st.session_state:
    st.session_state.distribution = {
        'publicSale': {'percentage': 20, 'tge': 10, 'cliff': 0, 'duration': 12},
        'privateRounds': {'percentage': 15, 'tge': 5, 'cliff': 6, 'duration': 24},
        'teamAndAdvisors': {'percentage': 15, 'tge': 0, 'cliff': 12, 'duration': 36},
        'development': {'percentage': 20, 'tge': 0, 'cliff': 6, 'duration': 48},
        'ecosystem': {'percentage': 15, 'tge': 5, 'cliff': 3, 'duration': 36},
        'treasury': {'percentage': 10, 'tge': 0, 'cliff': 12, 'duration': 48},
        'liquidityPool': {'percentage': 5, 'tge': 100, 'cliff': 0, 'duration': 0}
    }

# Title
st.title("🪙 Advanced Tokenomics Planner")

# Create two columns for the main layout
col1, col2 = st.columns([3, 2])

with col1:
    # Input fields
    total_supply = st.number_input("Total Supply", value=1000000000, step=1000000)
    initial_price = st.number_input("Initial Token Price ($)", value=0.001, format="%.6f")

    # Calculate key metrics
    fdv = total_supply * initial_price

    # Distribution settings
    st.subheader("Token Distribution")
    
    total_percentage = 0
    tge_circulating = 0

    for category, data in st.session_state.distribution.items():
        with st.expander(f"{category.replace('_', ' ').title()}"):
            percentage = st.slider(
                "Percentage",
                0.0, 100.0, float(data['percentage']),
                key=f"{category}_percentage"
            )
            
            # Update session state
            st.session_state.distribution[category]['percentage'] = percentage
            
            # Calculate tokens
            tokens = (percentage / 100) * total_supply
            st.write(f"Tokens: {tokens:,.0f}")
            
            # TGE and vesting settings
            cols = st.columns(3)
            with cols[0]:
                tge = st.number_input("TGE %", 0, 100, data['tge'], key=f"{category}_tge")
                st.session_state.distribution[category]['tge'] = tge
            with cols[1]:
                cliff = st.number_input("Cliff (months)", 0, 48, data['cliff'], key=f"{category}_cliff")
                st.session_state.distribution[category]['cliff'] = cliff
            with cols[2]:
                duration = st.number_input("Duration (months)", 0, 48, data['duration'], key=f"{category}_duration")
                st.session_state.distribution[category]['duration'] = duration
            
            total_percentage += percentage
            tge_circulating += (tokens * tge / 100)

    # Progress bar for total allocation
    st.subheader("Total Allocation")
    progress_color = "red" if total_percentage > 100 else "green"
    st.progress(min(total_percentage / 100, 1.0), text=f"{total_percentage:.1f}%")
    if total_percentage > 100:
        st.error("Total allocation exceeds 100%")

with col2:
    # Key metrics
    st.subheader("Key Metrics")
    metrics_cols = st.columns(2)
    
    with metrics_cols[0]:
        st.metric("Initial Market Cap", f"${(tge_circulating * initial_price):,.2f}")
        st.metric("TGE Circulating %", f"{(tge_circulating / total_supply * 100):.2f}%")
    
    with metrics_cols[1]:
        st.metric("Fully Diluted Valuation", f"${fdv:,.2f}")
        if tge_circulating > 0:
            st.metric("FDV/MCap Ratio", f"{(fdv / (tge_circulating * initial_price)):.2f}x")

    # Distribution Pie Chart
    distribution_data = pd.DataFrame([
        {"Category": k, "Percentage": v['percentage']}
        for k, v in st.session_state.distribution.items()
    ])
    
    fig_pie = px.pie(
        distribution_data,
        values='Percentage',
        names='Category',
        title='Token Distribution'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Unlock schedule calculation and visualization
months = range(49)  # 0 to 48 months
circulating_supply = np.zeros(len(months))

for category, data in st.session_state.distribution.items():
    tokens = (data['percentage'] / 100) * total_supply
    tge_amount = tokens * (data['tge'] / 100)
    remaining_amount = tokens - tge_amount
    monthly_unlock = remaining_amount / data['duration'] if data['duration'] > 0 else 0
    
    # Add TGE amount
    circulating_supply[0] += tge_amount
    
    # Add linear unlocks
    for month in range(1, len(months)):
        if month > data['cliff'] and month <= (data['cliff'] + data['duration']):
            circulating_supply[month] += monthly_unlock

# Calculate cumulative supply
cumulative_supply = np.cumsum(circulating_supply)
circulating_percentages = (cumulative_supply / total_supply) * 100

# Create unlock schedule chart
fig_unlock = go.Figure()
fig_unlock.add_trace(go.Scatter(
    x=months,
    y=circulating_percentages,
    mode='lines',
    name='Circulating Supply %'
))

fig_unlock.update_layout(
    title='Token Unlock Schedule',
    xaxis_title='Months after TGE',
    yaxis_title='Circulating Supply %',
    hovermode='x'
)

st.plotly_chart(fig_unlock, use_container_width=True)

# Display warnings
st.subheader("Warnings")
if (tge_circulating / total_supply * 100) > 25:
    st.warning("High TGE unlock may cause price instability")
if fdv / (tge_circulating * initial_price) > 100:
    st.warning("High FDV/MCap ratio indicates significant future dilution")
if st.session_state.distribution['teamAndAdvisors']['percentage'] > 20:
    st.warning("Team allocation appears high")
if st.session_state.distribution['liquidityPool']['percentage'] < 5:
    st.warning("Low liquidity allocation may cause price volatility")