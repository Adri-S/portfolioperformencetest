import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
import time
from scipy.optimize import minimize

# Setting up the page configuration
st.set_page_config(
    page_title="Interactive Sidebar",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'gauge_bgcolor' not in st.session_state:
    st.session_state.background_color_page="snow"
    st.session_state.chart_bgcolor="#ffffff"
    st.session_state.text_color="black"
    st.session_state.gauge_bgcolor="whitesmoke"
    st.session_state.border_color_input="#e9ecef"

custom_css = f""" 
<style>
    section[data-testid="stSidebar"],
    .stNumberInput input {{
        background-color: {st.session_state.chart_bgcolor};
        color: {st.session_state.text_color};
    }}
    header[data-testid="stHeader"] {{
        background-color: {st.session_state.background_color_page};
    }}
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stForm"],
    .st-cr {{
        border-color: {st.session_state.border_color_input};
    }}   
    .stButton button,
    .stTextInput input,
    .stDateInput input {{
        background-color: {st.session_state.chart_bgcolor};
        color: {st.session_state.text_color};
        border-color: {st.session_state.border_color_input};
    }}
    .stButton button {{
        height: 60px;
    }}
    div[data-testid="stNumberInput"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="InputInstructions"] {{
        color: {st.session_state.text_color};
    }}
    /* Ensure the table spans the full width of its container */
    .custom-table {{
        width: 100%;
        border-collapse: separate; /* Separate borders to allow for rounded corners */
        border-spacing: 0 15px; /* Space between rows */
        border-color: rgba(0,0,0,0);
        border: none; /* Remove any border around the table */
        vertical-align: center; /* Center text alignment vertically */
        font-size: 30px;
    }}
    /* Add space between rows */
    .custom-table th {{
        border-color: rgba(0,0,0,0);
        color: cornflowerblue;
        background-color: {st.session_state.chart_bgcolor};
        text-align: left; /* Center text alignment */
        padding-top: 20px;
        padding-bottom: 20px;
    }}
    .custom-table td {{
        background-color: {st.session_state.chart_bgcolor};
        border-color: rgba(0,0,0,0);
        text-align: right;
        color: {st.session_state.text_color};
        font-size: 30px;
        height: 60px;
        padding-top: 20px;
        padding-bottom: 20px;
    }}
    .custom-table td:nth-child(1),
    .custom-table td:nth-child(2) {{
        text-align: left;
    }}
    .custom-table td:first-child, th:first-child {{
        border-top-left-radius: 10px;
        border-bottom-left-radius: 10px;
    }}
    .custom-table td:last-child, th:last-child{{
        border-top-right-radius: 10px;
        border-bottom-right-radius: 10px;
    }}
    /* Make sure button styles are consistent */
    .stButton>button {{
        width: 100%;
    }}
    /* Ensure the body and main container have a consistent background color */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {{
        background-color: {st.session_state.get('background_color_page', 'snow')}; /* Use session state or default to snow */
    }}
    /* Ensure that dataframes use the full width */
    .pddataframe {{
        width: 100%;
    }}
</style>
"""

# Function to fetch stock data and handle non-trading days
def get_entry_price(ticker, purchase_date):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(start=purchase_date - datetime.timedelta(days=2), end=purchase_date + datetime.timedelta(days=2))
    if purchase_date in historical_data.index:
        return historical_data.loc[purchase_date]['Close']
    else:
        return historical_data['Close'].iloc[-1]

# Function to update the portfolio weights
def update_current_weights():
    total_value = sum(st.session_state.quantities[i] * st.session_state.current_prices[i] for i in range(len(st.session_state.stocks)))
    if total_value > 0:
        st.session_state.current_weights = [(st.session_state.quantities[i] * st.session_state.current_prices[i]) / total_value for i in range(len(st.session_state.stocks))]
    else:
        st.session_state.current_weights = [0] * len(st.session_state.stocks)

# Function to update the portfolio weights
def update_initial_weights():
    initial_total_value = sum(st.session_state.quantities[i] * st.session_state.bought_for[i] for i in range(len(st.session_state.stocks)))
    st.session_state.initial_weights = [(st.session_state.quantities[i] * st.session_state.bought_for[i]) / initial_total_value for i in range(len(st.session_state.stocks))]
        
# Initialize session state for the stock list
if 'stocks' not in st.session_state:
    # Pre-filled values for the portfolio
    initial_stocks = ["AAPL", "AMZN", "MSFT", "NVDA"]
    initial_quantities = [3, 4, 3, 2]
    st.session_state.initial_date = datetime.date(2023, 7, 30)  # Store initial date in session state

    # Fetch initial prices and current prices
    initial_prices = [get_entry_price(stock, st.session_state.initial_date) for stock in initial_stocks]
    initial_current_prices = [yf.Ticker(stock).history(period="1d")['Close'].iloc[0] for stock in initial_stocks]

    initial_last_30_days_return = []
    initial_last_90_days_return = []
    initial_last_180_days_return = []
    initial_last_365_days_return = []
    initial_total_gains = []

    for stock in initial_stocks:
        last_30_days = yf.Ticker(stock).history(period="1mo")['Close']
        last_90_days = yf.Ticker(stock).history(period="3mo")['Close']
        last_180_days = yf.Ticker(stock).history(period="6mo")['Close']
        last_365_days = yf.Ticker(stock).history(period="1y")['Close']
        
        # Calculate last month return
        if len(last_30_days) > 1:
            last_30_days_return = ((last_30_days.iloc[-1] - last_30_days.iloc[0]) / last_30_days.iloc[0]) * 100
        else:
            last_30_days_return = 0  # or handle as you prefer

        # Calculate last quarter return
        if len(last_90_days) > 1:
            last_90_days_return = ((last_90_days.iloc[-1] - last_90_days.iloc[0]) / last_90_days.iloc[0]) * 100
        else:
            last_90_days_return = 0  # or handle as you prefer

        # Calculate last quarter return
        if len(last_180_days) > 1:
            last_180_days_return = ((last_180_days.iloc[-1] - last_180_days.iloc[0]) / last_180_days.iloc[0]) * 100
        else:
            last_180_days_return = 0  # or handle as you prefer

        # Calculate last year return
        if len(last_365_days) > 1:
            last_365_days_return = ((last_365_days.iloc[-1] - last_365_days.iloc[0]) / last_365_days.iloc[0]) * 100
        else:
            last_365_days_return = 0  # or handle as you prefer
        
        # Calculate total gain
        total_gain = ((initial_current_prices[initial_stocks.index(stock)] - initial_prices[initial_stocks.index(stock)]) / initial_prices[initial_stocks.index(stock)]) * 100

        initial_last_30_days_return.append(last_30_days_return)
        initial_last_90_days_return.append(last_90_days_return)
        initial_last_180_days_return.append(last_180_days_return)
        initial_last_365_days_return.append(last_365_days_return)
        initial_total_gains.append(total_gain)

    total_value = sum(initial_quantities[i] * initial_current_prices[i] for i in range(len(initial_stocks)))
    current_weights = [(initial_quantities[i] * initial_current_prices[i]) / total_value for i in range(len(initial_stocks))]

    initial_total_value = sum(initial_quantities[i] * initial_prices[i] for i in range(len(initial_stocks)))
    initial_weights = [(initial_quantities[i] * initial_prices[i]) / initial_total_value for i in range(len(initial_stocks))]

    market_value_stocks = [initial_current_prices[i] * initial_quantities[i] for i in range(len(initial_stocks))]

    # Initialize all session state variables
    st.session_state.stocks = initial_stocks
    st.session_state.dates = [st.session_state.initial_date] * len(initial_stocks)
    st.session_state.bought_for = initial_prices
    st.session_state.quantities = initial_quantities
    st.session_state.current_prices = initial_current_prices
    st.session_state.last_30_days_return = initial_last_30_days_return
    st.session_state.last_90_days_return = initial_last_90_days_return
    st.session_state.last_180_days_return = initial_last_180_days_return
    st.session_state.last_365_days_return = initial_last_365_days_return
    st.session_state.total_gains = initial_total_gains
    st.session_state.current_weights = current_weights
    st.session_state.initial_weights = initial_weights
    st.session_state.initial_total_value = initial_total_value
    st.session_state.market_value_stocks = market_value_stocks


# Title
st.title("Welcome to Your Portfolio Page")

with st.form(key='add_stock_form'):

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_stock = st.text_input('Stock Ticker (e.g., AAPL)', key="new stock", max_chars=6)
    with col2:
        new_quantity = st.number_input('Quantity', min_value=1, max_value=1000, step=1, key="new_quantity")
    with col3:
        purchase_date = st.date_input('Purchase Date', key="date", value=st.session_state.dates[0] if st.session_state.dates else datetime.datetime.now().date())

    if len(st.session_state.stocks) < 10:
        if st.form_submit_button("Add to Portfolio"):
            if new_stock and new_quantity:
                # Fetch the stock data to get the entry price
                entry_price = get_entry_price(new_stock, purchase_date)

                # Fetch current price
                current_price = yf.Ticker(new_stock).history(period="1d")['Close'].iloc[0]

                # Calculate returns
                last_30_days = yf.Ticker(new_stock).history(period="1mo")['Close']
                last_90_days = yf.Ticker(new_stock).history(period="3mo")['Close']
                last_180_days = yf.Ticker(new_stock).history(period="6mo")['Close']
                last_365_days = yf.Ticker(new_stock).history(period="1y")['Close']
                if not last_30_days.empty and len(last_30_days) > 1:
                    last_30_days_return = ((last_30_days.iloc[-1] - last_30_days.iloc[0]) / last_30_days.iloc[0]) * 100
                else:
                    last_30_days_return = None  # Or some default value or handling
                    print("Insufficient data to calculate return")
                if not last_90_days.empty and len(last_90_days) > 1:
                    last_90_days_return = ((last_30_days.iloc[-1] - last_90_days.iloc[0]) / last_90_days.iloc[0]) * 100
                else:
                    last_90_days_return = None  # Or some default value or handling
                    print("Insufficient data to calculate return")
                if not last_180_days.empty and len(last_180_days) > 1:
                    last_180_days_return = ((last_180_days.iloc[-1] - last_180_days.iloc[0]) / last_180_days.iloc[0]) * 100
                else:
                    last_180_days_return = None  # Or some default value or handling
                    print("Insufficient data to calculate return")
                if not last_365_days.empty and len(last_365_days) > 1:
                    last_365_days_return = ((last_365_days.iloc[-1] - last_365_days.iloc[0]) / last_365_days.iloc[0]) * 100
                else:
                    last_365_days_return = None  # Or some default value or handling
                    print("Insufficient data to calculate return")
                total_gain = ((current_price - entry_price) / entry_price) * 100
                market_value_stocks = current_price * new_quantity
                
                st.session_state.stocks.append(new_stock)
                st.session_state.dates.append(purchase_date)  # Same start date for all
                st.session_state.bought_for.append(entry_price)
                st.session_state.quantities.append(new_quantity)
                st.session_state.current_prices.append(current_price)
                st.session_state.last_30_days_return.append(last_30_days_return)
                st.session_state.last_90_days_return.append(last_90_days_return)
                st.session_state.last_180_days_return.append(last_180_days_return)
                st.session_state.last_365_days_return.append(last_365_days_return)
                st.session_state.total_gains.append(total_gain)
                st.session_state.market_value_stocks.append(market_value_stocks)

                update_current_weights()
                update_initial_weights()

                st.rerun()
    else:
        st.form_submit_button("+ add stock", disabled=True,)

    if len(st.session_state.stocks) == 10:
        st.warning("For simplification you can only hold a maximum of 10 stocks. Please remove a stock before you add a new one.")

# Create the data for the table
positive_stocks = ["#47B382", "#e5f2e9" ] # text-color, background-color
negative_stocks = ["#e62a3f", "#fcded8"] # text-color, background-color

data = {
    "Stock": st.session_state.stocks,
    "Date": st.session_state.dates,
    "Bought for": [f"{value:,.2f} $" for value in st.session_state.bought_for],
    "Quantity": st.session_state.quantities,
    "Current Price": [f"{value:,.2f} $" for value in st.session_state.current_prices],
    "Market Value": [f"{value:,.2f} $" for value in st.session_state.market_value_stocks],
    "30 Days": [f'<span style="background-color: {positive_stocks[1] if value >= 0 else negative_stocks[1]}; border-radius: 12px; padding: 8px; color: {positive_stocks[0] if value >= 0 else negative_stocks[0]};">{"+" if value >= 0 else "-"} {abs(round(value, 2))}%</span>' 
    for value in st.session_state.last_30_days_return],
    "90 Days": [f'<span style="background-color: {positive_stocks[1] if value >= 0 else negative_stocks[1]}; border-radius: 12px; padding: 8px; color: {positive_stocks[0] if value >= 0 else negative_stocks[0]};">{"+" if value >= 0 else "-"} {abs(round(value, 2))}%</span>'
    for value in st.session_state.last_90_days_return],
    "180 Days": [f'<span style="background-color: {positive_stocks[1] if value >= 0 else negative_stocks[1]}; border-radius: 12px; padding: 8px; color: {positive_stocks[0] if value >= 0 else negative_stocks[0]};">{"+" if value >= 0 else "-"} {abs(round(value, 2))}%</span>'
    for value in st.session_state.last_180_days_return],
    "365 Days": [f'<span style="background-color: {positive_stocks[1] if value >= 0 else negative_stocks[1]}; border-radius: 12px; padding: 8px; color: {positive_stocks[0] if value >= 0 else negative_stocks[0]};">{"+" if value >= 0 else "-"} {abs(round(value, 2))}%</span>'
    for value in st.session_state.last_365_days_return],
    "Total gain": [f'<span style="background-color: {positive_stocks[1] if value >= 0 else negative_stocks[1]}; border-radius: 12px; padding: 8px; color: {positive_stocks[0] if value >= 0 else negative_stocks[0]};">{"+" if value >= 0 else "-"} {abs(round(value, 2))}%</span>'
    for value in st.session_state.total_gains],
#    "Weight in Portfolio (Initial)": [
#        f"{(st.session_state.quantities[i] * st.session_state.bought_for[i]) / sum(st.session_state.quantities[j] * st.session_state.bought_for[j] for j in range(len(st.session_state.stocks))) * 100:.2f}%"
#        for i in range(len(st.session_state.stocks))
#    ],
    "Current Weight": [
        f'<div style="display: flex; align-items: center;">' +
        f'<div style="width: 40px; height: 40px; position: relative; flex-shrink: 0; margin-left: 30px;">' +
        f'<svg viewBox="0 0 40 40" style="transform: rotate(-90deg);">' +
        f'<circle cx="20" cy="20" r="15.91549431" fill="transparent" stroke-width="5" stroke={st.session_state.border_color_input} />' +
        f'<circle cx="20" cy="20" r="15.91549431" fill="transparent" stroke-width="5" ' +
        f'stroke-dasharray="{(st.session_state.quantities[i] * st.session_state.current_prices[i]) / sum(st.session_state.quantities[j] * st.session_state.current_prices[j] for j in range(len(st.session_state.stocks))) * 100:.2f}, 100" ' +
        f'stroke="gray" />' +
        f'</svg>' +
        f'</div>' +
        f'<span style="margin-left: auto; margin-right: 8px;">{(st.session_state.quantities[i] * st.session_state.current_prices[i]) / sum(st.session_state.quantities[j] * st.session_state.current_prices[j] for j in range(len(st.session_state.stocks))) * 100:.2f}%</span>' +
        f'</div>'
        for i in range(len(st.session_state.stocks))
    ],
}

df = pd.DataFrame(data)
df = df.round(2)

# Convert the DataFrame to HTML with custom class
custom_table_html = df.to_html(classes='custom-table', index=False, escape=False)

# Inject the CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Display the custom styled table
st.markdown(custom_table_html, unsafe_allow_html=True)

# Button to remove last stock
if st.button("Remove Last Stock", key="remove_stock_button"):
    if st.session_state.stocks:
        st.session_state.stocks.pop()
        st.session_state.dates.pop()
        st.session_state.bought_for.pop()
        st.session_state.quantities.pop()
        st.session_state.current_prices.pop()
        st.session_state.last_30_days_return.pop()
        st.session_state.last_90_days_return.pop()
        st.session_state.last_180_days_return.pop()
        st.session_state.last_365_days_return.pop()
        st.session_state.total_gains.pop()
        st.session_state.current_weights.pop()
        st.session_state.initial_weights.pop()
        st.session_state.market_value_stocks.pop()

        # Recalculate portfolio weights
        update_current_weights()
        update_initial_weights()
        st.rerun()

if 'rerun_triggered' in st.session_state:
    del st.session_state.rerun_triggered

if len(st.session_state.stocks) < 3:
    st.warning("You need to at least add 3 stocks to your portfolio in order for it to work.")
    st.stop()
    
# Monte Carlo Simulation

# risk free rate on base of the 13 week US treasury bill
start_date_cpi = datetime.datetime.today() - datetime.timedelta(days=365)
treasury_data = yf.download('^IRX', start=start_date_cpi)
risk_free_rate = treasury_data.iloc[0,3]
st.session_state.risk_free_rate = risk_free_rate
st.session_state.risk_free_daily_rate = risk_free_rate/252/100

# Calculate Fixed Weight Portfolio performance
stock_price_data = yf.download(st.session_state.stocks, start=datetime.datetime.today() - datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
stock_price_data = stock_price_data[st.session_state.stocks]
stock_volume_data = stock_price_data * st.session_state.quantities
total_volume = stock_volume_data.sum(axis=1)
portfolio_annual_return = (total_volume.iloc[-1] - total_volume.iloc[0]) / total_volume.iloc[0] * 100
portfolio_returns_daily = total_volume.pct_change()
portfolio_vola = portfolio_returns_daily.var()
portfolio_std = np.sqrt(portfolio_vola) * 252 ** 0.5

selected = st.session_state.stocks
mc_initial_weights = [(st.session_state.quantities[i] * st.session_state.bought_for[i]) / sum(st.session_state.quantities[j] * st.session_state.bought_for[j] for j in range(len(st.session_state.stocks))) for i in range(len(st.session_state.stocks))]
start_date = st.session_state.dates[0]
end_date = datetime.datetime.now().date()

# download adjusted closing prices of the selected companies
data = yf.download(selected, start=start_date, end=end_date)['Adj Close']

# calculate daily and annual returns of the stocks
returns_daily = data.pct_change()
returns_annual = returns_daily.mean() * len(returns_daily)

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = returns_daily.cov() * len(returns_daily)

# empty lists to store returns, volatility and weights of imaginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
num_portfolios = 50000

# Calculate the returns and volatility for the fixed weight portfolio
fixed_weights = np.array(mc_initial_weights)
fixed_return = portfolio_annual_return/100
fixed_volatility = portfolio_std
st.session_state.mc_fixret = fixed_return
st.session_state.mc_fixvol = fixed_volatility

# set random seed for reproduction's sake
np.random.seed(101)

# Populate the empty lists with each portfolio's returns, risk, and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    returns = np.dot(weights, returns_annual)
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accommodate each ticker and weight in the portfolio
for counter, symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]

# reorder dataframe columns
df = df[column_order]

st.session_state.mc_allret = df['Returns']
st.session_state.mc_allvol = df['Volatility']
st.session_state.mc_all_sharpe = df['Sharpe Ratio']

# Function to calculate portfolio performance
def portfolio_performance(weights, returns_annual, cov_annual):
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    return returns, volatility

# Function to minimize volatility for a given return
def minimize_volatility(returns_annual, cov_annual, target_return):
    num_assets = len(returns_annual)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, returns_annual, cov_annual)[0] - target_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(lambda x: portfolio_performance(x, returns_annual, cov_annual)[1], 
                      num_assets * [1. / num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Calculate the efficient frontier
target_returns = np.linspace(df['Returns'].min(), df['Returns'].max(), 100)
efficient_volatilities = []
efficient_weights = []

for target_return in target_returns:
    efficient_portfolio = minimize_volatility(returns_annual, cov_annual, target_return)
    efficient_volatilities.append(efficient_portfolio.fun)
    efficient_weights.append(efficient_portfolio.x)
st.session_state.mc_effret = target_returns
st.session_state.mc_effvol = efficient_volatilities

# Find the minimum variance portfolio
min_variance_index = np.argmin(efficient_volatilities)
min_variance_volatility = efficient_volatilities[min_variance_index]
min_variance_return = target_returns[min_variance_index]
st.session_state.mc_minret = min_variance_return
st.session_state.mc_minvol = min_variance_volatility

# Ensure the portfolio with the lowest volatility for the same or higher return is not below the minimum variance portfolio
valid_indices = [i for i, ret in enumerate(target_returns) if ret >= min_variance_return]
valid_volatilities = [efficient_volatilities[i] for i in valid_indices]
valid_returns = [target_returns[i] for i in valid_indices]
same_return_index = (np.abs(np.array(valid_returns) - fixed_return)).argmin()
lowest_vol_same_return = (valid_returns[same_return_index], valid_volatilities[same_return_index], efficient_weights[same_return_index])
st.session_state.mc_lowvol = lowest_vol_same_return[1]
st.session_state.mc_lowret = lowest_vol_same_return[0]

# Find the portfolio with the highest expected return for the closest level of volatility to the fixed weight portfolio
for i, (ret, vol) in enumerate(zip(target_returns, efficient_volatilities)):
    if ret >= fixed_return and vol <= fixed_volatility:
        highest_return = ret
        highest_return_same_vol = (ret, vol, efficient_weights[i])
st.session_state.mc_higretsamvol = highest_return_same_vol
st.session_state.mc_highvol = highest_return_same_vol[1]
st.session_state.mc_highret = highest_return_same_vol[0]

# Function to update portfolio weights
def update_weights_minvar():
    st.session_state.total_value = sum(st.session_state.quantities[i] * st.session_state.current_prices[i] for i in range(len(st.session_state.stocks)))
    if st.session_state.total_value > 0:
        st.session_state.weights = [(st.session_state.quantities[i] * st.session_state.current_prices[i]) / st.session_state.total_value for i in range(len(st.session_state.stocks))]
    else:
        st.session_state.weights = [0] * len(st.session_state.stocks)

# Update weights based on new quantities
update_weights_minvar()
highest_weights = highest_return_same_vol[2]
st.session_state.weights_highret = highest_weights
lowest_weights = lowest_vol_same_return[2]
st.session_state.weights_minvar = lowest_weights

st.session_state.update({
    'weights_minvar': lowest_weights,
    'weights_highret': highest_weights,
})

st.success("You are ready to go! Explore the KPIs page")
