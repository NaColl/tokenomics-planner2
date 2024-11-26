import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from decimal import Decimal, ROUND_DOWN

# Utility functions for safe calculations
def safe_percentage(value, total):
    try:
        percentage = (value / total * 100) if total != 0 else 0
        return round(percentage, 2)
    except (TypeError, ZeroDivisionError):
        return 0.0

def safe_division(numerator, denominator):
    try:
        return numerator / denominator if denominator != 0 else 0
    except (TypeError, ZeroDivisionError):
        return 0

def format_large_number(num):
    try:
        if num >= 1_000_000_000:
            return f"${num/1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"${num/1_000:.2f}K"
        else:
            return f"${num:.2f}"
    except (TypeError, ValueError):
        return "$0.00"

st.set_page_config(page_title="Tokenomics Planner", layout="wide")

# Initialize session state with validated defaults
if 'distribution' not in st.session_state:
    st.session_state.distribution = {
        'publicSale': {'percentage': 20.0, 'tge': 10.0, 'cliff': 0, 'duration': 12},
        'privateRounds': {'percentage': 15.0, 'tge': 5.0, 'cliff': 6, 'duration': 24},
        'teamAndAdvisors': {'percentage': 15.0, 'tge': 0.0, 'cliff': 12, 'duration': 36},
        'development': {'percentage': 20.0, 'tge': 0.0, 'cliff': 6, 'duration': 48},
        'ecosystem': {'percentage': 15.0, 'tge': 5.0, 'cliff': 3, 'duration': 36},
        'treasury': {'percentage': 10.0, 'tge': 0.0, 'cliff': 12, 'duration': 48},
        'liquidityPool': {'percentage': 5.0, 'tge': 100.0, 'cliff': 0, 'duration': 0}
    }

if 'remaining_percentage' not in st.session_state:
    st.session_state.remaining_percentage = 100.0

st.title("🪙 Advanced Tokenomics Planner")

col1, col2 = st.columns([3, 2])

with col1:
    # Input fields with strict validation
    total_supply = st.number_input(
        "Total Supply",
        min_value=1.0,
        max_value=1_000_000_000_000.0,  # 1 trillion max supply
        value=1_000_000_000.0,
        step=1_000_000.0,
        format="%.0f"
    )
    
    initial_price = st.number_input(
        "Initial Token Price ($)",
        min_value=0.0000001,
        max_value=1_000_000.0,
        value=0.001,
        format="%.8f"
    )

    # Calculate key metrics using Decimal for precision
    fdv = Decimal(str(total_supply)) * Decimal(str(initial_price))
    fdv = float(fdv.quantize(Decimal('0.00'), rounding=ROUND_DOWN))

    st.subheader("Token Distribution")
    
    total_percentage = Decimal('0.0')
    tge_circulating = Decimal('0.0')
    st.session_state.remaining_percentage = Decimal('100.0')

    for category, data in st.session_state.distribution.items():
        with st.expander(f"{category.replace('_', ' ').title()}"):
            current_percentage = Decimal(str(data['percentage']))
            other_percentages = sum(Decimal(str(v['percentage'])) for k, v in st.session_state.distribution.items() if k != category)
            max_allowed = max(Decimal('0.0'), min(Decimal('100.0'), Decimal('100.0') - other_percentages))
            
            percentage = st.slider(
                "Percentage",
                min_value=float(Decimal('0.0')),
                max_value=float(max_allowed),
                value=float(min(current_percentage, max_allowed)),
                step=0.1,
                key=f"{category}_percentage"
            )
            
            st.session_state.distribution[category]['percentage'] = float(percentage)
            
            # Token calculations with Decimal
            tokens = (Decimal(str(percentage)) / Decimal('100.0')) * Decimal(str(total_supply))
            tokens = tokens.quantize(Decimal('1.'), rounding=ROUND_DOWN)
            st.write(f"Tokens: {float(tokens):,.0f}")
            
            cols = st.columns(3)
            with cols[0]:
                tge = st.number_input(
                    "TGE %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(data['tge']),
                    key=f"{category}_tge"
                )
                st.session_state.distribution[category]['tge'] = tge
            with cols[1]:
                cliff = st.number_input(
                    "Cliff (months)",
                    min_value=0,
                    max_value=48,
                    value=int(data['cliff']),
                    key=f"{category}_cliff"
                )
                st.session_state.distribution[category]['cliff'] = cliff
            with cols[2]:
                duration = st.number_input(
                    "Duration (months)",
                    min_value=0,
                    max_value=48,
                    value=int(data['duration']),
                    key=f"{category}_duration"
                )
                st.session_state.distribution[category]['duration'] = duration
            
            total_percentage += Decimal(str(percentage))
            tge_amount = tokens * (Decimal(str(tge)) / Decimal('100.0'))
            tge_circulating += tge_amount.quantize(Decimal('1.'), rounding=ROUND_DOWN)

    st.subheader("Total Allocation")
    progress_color = "red" if total_percentage > Decimal('100.0') else "green"
    st.progress(min(float(total_percentage / Decimal('100.0')), 1.0), text=f"{float(total_percentage):.1f}%")
    if total_percentage > Decimal('100.0'):
        st.error("Total allocation exceeds 100%")
    elif total_percentage < Decimal('100.0'):
        st.warning(f"Remaining allocation: {float(Decimal('100.0') - total_percentage):.1f}%")

with col2:
    st.subheader("Key Metrics")
    metrics_cols = st.columns(2)
    
    with metrics_cols[0]:
        initial_mcap = float(tge_circulating) * float(initial_price)
        st.metric("Initial Market Cap", format_large_number(initial_mcap))
        tge_percentage = safe_percentage(float(tge_circulating), float(total_supply))
        st.metric("TGE Circulating %", f"{tge_percentage:.2f}%")
    
    with metrics_cols[1]:
        st.metric("Fully Diluted Valuation", format_large_number(fdv))
        if float(tge_circulating) > 0:
            fdv_mcap_ratio = safe_division(fdv, initial_mcap)
            st.metric("FDV/MCap Ratio", f"{fdv_mcap_ratio:.2f}x")

    distribution_data = pd.DataFrame([
        {"Category": k.replace('_', ' ').title(), "Percentage": v['percentage']}
        for k, v in st.session_state.distribution.items()
        if v['percentage'] > 0
    ])
    
    fig_pie = px.pie(
        distribution_data,
        values='Percentage',
        names='Category',
        title='Token Distribution'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Vesting schedule calculations with improved precision
months = np.arange(49, dtype=np.int32)
circulating_supply = np.zeros(len(months), dtype=np.float64)

for category, data in st.session_state.distribution.items():
    tokens = (Decimal(str(data['percentage'])) / Decimal('100.0')) * Decimal(str(total_supply))
    tokens = tokens.quantize(Decimal('1.'), rounding=ROUND_DOWN)
    
    tge_amount = tokens * (Decimal(str(data['tge'])) / Decimal('100.0'))
    tge_amount = tge_amount.quantize(Decimal('1.'), rounding=ROUND_DOWN)
    
    remaining_amount = tokens - tge_amount
    monthly_unlock = (remaining_amount / Decimal(str(data['duration']))) if data['duration'] > 0 else Decimal('0')
    monthly_unlock = monthly_unlock.quantize(Decimal('1.'), rounding=ROUND_DOWN)
    
    circulating_supply[0] += float(tge_amount)
    
    for month in range(1, len(months)):
        if month > data['cliff'] and month <= (data['cliff'] + data['duration']):
            circulating_supply[month] += float(monthly_unlock)

cumulative_supply = np.zeros(len(months), dtype=np.float64)
for i in range(len(months)):
    if i == 0:
        cumulative_supply[i] = circulating_supply[i]
    else:
        cumulative_supply[i] = cumulative_supply[i-1] + circulating_supply[i]

circulating_percentages = np.array([safe_percentage(supply, float(total_supply)) for supply in cumulative_supply])

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
    hovermode='x',
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig_unlock, use_container_width=True)

# Warning system with improved calculations
st.subheader("Warnings")

if tge_percentage > 25:
    st.warning(f"High TGE unlock ({tge_percentage:.1f}%) may cause price instability")

if float(tge_circulating) > 0:
    fdv_mcap_ratio = safe_division(fdv, initial_mcap)
    if fdv_mcap_ratio > 100:
        st.warning(f"High FDV/MCap ratio ({fdv_mcap_ratio:.1f}x) indicates significant future dilution")

team_percentage = st.session_state.distribution['teamAndAdvisors']['percentage']
if team_percentage > 20:
    st.warning(f"Team allocation ({team_percentage:.1f}%) appears high")

liquidity_percentage = st.session_state.distribution['liquidityPool']['percentage']
if liquidity_percentage < 5:
    st.warning(f"Low liquidity allocation ({liquidity_percentage:.1f}%) may cause price volatility")
