import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import requests
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

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

if 'weights_minvar' not in st.session_state:
    st.warning("Please start at our portfolio side.")
    st.stop()

if len(st.session_state.stocks) < 2:
    st.warning("You need to at least add 2 stocks to your portfolio in order for it to work.")
    st.stop()

# CSS for general styling
st.markdown(f"""
    <!-- Bootstrap Icons CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">

    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {{
            background-color: {st.session_state.background_color_page};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {st.session_state.chart_bgcolor};
        }}
        header[data-testid="stHeader"] {{
            background-color: {st.session_state.background_color_page};
            color: {st.session_state.text_color};
        }}
        .container1 {{
            padding: 15px;
        }}
        .bi {{
            display: inline-block;
            font-size: 30;
            vertical-align: middle;
            color: yellow;
        }}
        .stPlotlyChart {{
            border-radius: 15px;
            overflow: hidden;
        }}
        .column-box-stocks-1 {{
            background-color: {st.session_state.chart_bgcolor};
            border-radius: 15px;
            color: {st.session_state.text_color};
            font-size: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            height: 100px; /* Increased height for the boxes */
        }}
        .column-box-stocks-2 {{
            background-color: {st.session_state.chart_bgcolor};
            border-radius: 15px;
            color: {st.session_state.text_color};
            font-size: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            height: 400px; /* Increased height for the boxes */
        }}
        .header-text {{
            position: absolute;
            font-weight: bold;
            font-size: 35px;
            color: cornflowerblue;
        }}
        .subheading-stocks {{
            position: absolute;
            font-weight: normal;
            font-size: 25px;
            color: {st.session_state.text_color};
        }}
        .stocks-top-numbers {{
            position: absolute;
            font-weight: bold;
            font-size: 30px;
            color: {st.session_state.text_color};
        }}
        .header-text {{ top: 10px; left: 30px; }}
        .subheading-stocks {{ top: 10px; right: 10px; }}
        .stocks-top-numbers {{ bottom: 10px; right: 10px; }}
    </style>
""", unsafe_allow_html=True)

# Set start date as the purchase date of the stocks
start_date = datetime.datetime.today() - datetime.timedelta(days=365)
end_date = datetime.datetime.now().date()

# Fetch MSCI World Index data
msci_world_data = yf.download('URTH', start=datetime.datetime.today()- datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
msci_world_data_harmonized = msci_world_data / msci_world_data.iloc[0] * 100 # Normalize the data
msci_world_annual_return = (msci_world_data.iloc[-1] - msci_world_data.iloc[0]) / msci_world_data.iloc[0] * 100
msci_returns_daily = msci_world_data.pct_change()
msci_world_vola = msci_returns_daily.var()
msci_world_sharpe_ratio = (msci_returns_daily.mean() - st.session_state.risk_free_daily_rate) / np.sqrt(msci_world_vola) * 252 ** 0.5
msci_world_sharpe = msci_world_data ** msci_world_sharpe_ratio
msci_world_data_sharpe = msci_world_sharpe / msci_world_sharpe.iloc[0] * 100

# Fetch iShares US Tech data
iShares_tech_data = yf.download('IYW', start=datetime.datetime.today()- datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
iShares_tech_data_harmonized = iShares_tech_data / iShares_tech_data.iloc[0] * 100 # Normalize the data
iShares_tech_returns_daily = iShares_tech_data.pct_change()
iShares_tech_vola = iShares_tech_returns_daily.var()
iShares_tech_sharpe_ratio = (iShares_tech_returns_daily.mean() - st.session_state.risk_free_daily_rate) / np.sqrt(iShares_tech_vola) * 252 ** 0.5
iShares_tech_sharpe = iShares_tech_data ** iShares_tech_sharpe_ratio
iShares_tech_data_sharpe = iShares_tech_sharpe / iShares_tech_sharpe.iloc[0] * 100

# Fetch iShares MSCI Emerging Markets data
iShares_eem_data = yf.download('EEM', start=datetime.datetime.today()- datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
iShares_eem_data_harmonized = iShares_eem_data / iShares_eem_data.iloc[0] * 100 # Normalize the data
iShares_eem_returns_daily = iShares_eem_data.pct_change()
iShares_eem_vola = iShares_eem_returns_daily.var()
iShares_eem_sharpe_ratio = (iShares_eem_returns_daily.mean() - st.session_state.risk_free_daily_rate) / np.sqrt(iShares_eem_vola) * 252 ** 0.5
iShares_eem_sharpe = iShares_eem_data ** iShares_eem_sharpe_ratio
iShares_eem_data_sharpe = iShares_eem_sharpe / iShares_eem_sharpe.iloc[0] * 100

# Calculate Fixed Weight Portfolio performance
stock_price_data = yf.download(st.session_state.stocks, start=start_date, end=end_date)['Adj Close']
stock_price_data = stock_price_data[st.session_state.stocks]
stock_volume_data = stock_price_data * st.session_state.quantities
total_volume = stock_volume_data.sum(axis=1)
portfolio_data_harmonized = total_volume / total_volume.iloc[0] * 100 # Normalize the data
portfolio_annual_return = (total_volume.iloc[-1] - total_volume.iloc[0]) / total_volume.iloc[0] * 100
portfolio_returns_daily = total_volume.pct_change()
portfolio_vola = portfolio_returns_daily.var()
portfolio_sharpe_ratio = (portfolio_returns_daily.mean() - st.session_state.risk_free_daily_rate) / np.sqrt(portfolio_vola) * 252 ** 0.5
portfolio_sharpe = total_volume ** portfolio_sharpe_ratio
portfolio_data_sharpe = portfolio_sharpe / portfolio_sharpe.iloc[0] * 100

def get_dividends(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    return dividends

def total_return_with_reinvestment(df):
    """
    Calculate the "Total Return" of a stock when dividends are
    reinvested in the stock.
    """
    df['Dividends'] = df['Dividends'].fillna(0)
    tot_ret_daily = (df['Dividends'] + df['Close']) / df['Close'].shift(1)
    tot_ret = tot_ret_daily.cumprod()
    tot_ret.iloc[0] = 1.0
    return tot_ret

portfolio_df = pd.DataFrame({
    "Stock": st.session_state.stocks,
    "Bought for": st.session_state.bought_for,
    "Quantity": st.session_state.quantities,
    "Current Price": st.session_state.current_prices,
    "Weight": st.session_state.current_weights,
    "Initial Weight": st.session_state.initial_weights,
    "30 Days": st.session_state.last_30_days_return,
    "90 Days": st.session_state.last_90_days_return,
    "180 Days": st.session_state.last_180_days_return,
    "365 Days": st.session_state.last_365_days_return,
})

portfolio_df['Return'] = (portfolio_df['Current Price'] - portfolio_df['Bought for']) * portfolio_df['Quantity']
portfolio_df['Market Value'] = portfolio_df['Current Price'] * portfolio_df['Quantity']

portfolio_total_return = portfolio_df['Return'].sum()
st.session_state.portfolio_total_return = "{:0,.2f}".format(portfolio_total_return)

portfolio_market_value = portfolio_df['Market Value'].sum()
st.session_state.portfolio_market_value = "{:0,.2f}".format(portfolio_market_value)

portfolio_relative_return = portfolio_total_return / portfolio_market_value
st.session_state.portfolio_relative_return = round(portfolio_relative_return, 2)

portfolio_df['30 Days'] = portfolio_df['30 Days'] * portfolio_df['Initial Weight']
portfolio_30_days_return = "{:0,.2f}".format(portfolio_df['30 Days'].sum())
portfolio_df['90 Days'] = portfolio_df['90 Days'] * portfolio_df['Initial Weight']
portfolio_90_days_return = "{:0,.2f}".format(portfolio_df['90 Days'].sum())
portfolio_df['180 Days'] = portfolio_df['180 Days'] * portfolio_df['Initial Weight']
portfolio_180_days_return = "{:0,.2f}".format(portfolio_df['180 Days'].sum())
portfolio_df['365 Days'] = portfolio_df['365 Days'] * portfolio_df['Initial Weight']
portfolio_365_days_return = "{:0,.2f}".format(portfolio_df['365 Days'].sum())

# Update session state pie chart data
st.session_state.pie_chart_data = {
    'labels': portfolio_df['Stock'].tolist(),
    'values': portfolio_df['Market Value'].tolist()
}

# Fetch MSCI World Index data
msci_world_data_gauge = yf.download('URTH', start=datetime.datetime.today()- datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
msci_world_data_gauge_number = (msci_world_data_gauge.iloc[-1] - msci_world_data_gauge.iloc[0]) / msci_world_data_gauge.iloc[0] * 100

# inflation rate on base of consumer price index
def fetch_cpi_data():
    start = datetime.datetime.today() - datetime.timedelta(days=430)
    end = datetime.datetime.today()
    cpi_data = web.DataReader('CPIAUCSL', 'fred', start, end)
    return cpi_data
cpi_data = fetch_cpi_data()
#st.write(cpi_data)
cpi_latest = cpi_data.iloc[11,0]
#st.write(cpi_latest)
cpi_one_year_ago = cpi_data.iloc[0,0]
#st.write(cpi_one_year_ago)
inflation_rate = ((cpi_latest - cpi_one_year_ago) / cpi_one_year_ago) * 100

# Title
st.title("Welcome to Analysis Page")

with st.container():
    
    portfolio_value, all_time_gain, past_30_days, past_90_days, past_180_days = st.columns(5, gap="medium")
    
    with portfolio_value:
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: #ffba08; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="bi bi-wallet2" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">Portfolio Value:</div>
                <div class="corner-text stocks-top-numbers">{st.session_state.portfolio_market_value} $</div>
            </div>
        """, unsafe_allow_html=True)

    with all_time_gain:
        portfolio_total_return_float = float(st.session_state.portfolio_total_return.replace(',', ''))
        if portfolio_total_return_float >= 0:
            ptr_bgc = "#62BD7D"
            ptr_icon = "bi bi-plus-circle-dotted"
            gain_loss = "gain"
        else:
            ptr_bgc = "#C84455"
            ptr_icon = "bi bi-dash-circle-dotted" 
            gain_loss = "loss"     
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: {ptr_bgc}; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="{ptr_icon}" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">all time {gain_loss}:</div>
                <div class="corner-text stocks-top-numbers">{st.session_state.portfolio_total_return} $</div>
            </div>
        """, unsafe_allow_html=True)

    with past_30_days:
        if float(portfolio_30_days_return) >= 0:
            tdy_bgc = "#62BD7D"
            tdy_icon = "bi bi-graph-up-arrow"
        else:
            tdy_bgc = "#C84455"
            tdy_icon = "bi bi-graph-down-arrow"
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: {tdy_bgc}; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="{tdy_icon}" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">past 30 days:</div>
                <div class="corner-text stocks-top-numbers">{portfolio_30_days_return} %</div>
            </div>
        """, unsafe_allow_html=True)

    with past_90_days:
        if float(portfolio_90_days_return) >= 0:
            fda_bgc = "#62BD7D"
            fda_icon = "bi bi-graph-up-arrow"
        else:
            fda_bgc = "#C84455"
            fda_icon = "bi bi-graph-down-arrow"
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: {fda_bgc}; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="{fda_icon}" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">past 90 days:</div>
                <div class="corner-text stocks-top-numbers">{portfolio_90_days_return} %</div>
            </div>
        """, unsafe_allow_html=True)

    with past_180_days:
        if float(portfolio_180_days_return) >= 0:
            fda_bgc = "#62BD7D"
            fda_icon = "bi bi-graph-up-arrow"
        else:
            fda_bgc = "#C84455"
            fda_icon = "bi bi-graph-down-arrow"
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: {fda_bgc}; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="{fda_icon}" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">past 180 days:</div>
                <div class="corner-text stocks-top-numbers">{portfolio_180_days_return} %</div>
            </div>
        """, unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="container1">', unsafe_allow_html=True)
    
    portfolio_revenue_chart, sharpe_loss, portfolio_consumption = st.columns([1, 1, 2], gap="medium")       

    with portfolio_revenue_chart:

        if portfolio_annual_return >= 0:
            fig_portfolio_revenue = go.Figure(go.Indicator(
                value=portfolio_annual_return,
                mode="gauge+number", # and not gauge+number+delta"
                number_suffix="%",
                gauge=dict(
                    axis=dict(
                        range=[0, max(portfolio_annual_return, inflation_rate, st.session_state.risk_free_rate, msci_world_annual_return)+1],
                        tickwidth=2, tickcolor="darkgray", ticksuffix="%",
                        tickfont=dict(color=st.session_state.text_color, size=20),
                    ),
                    bar=dict(color="cornflowerblue"),
                    bordercolor=st.session_state.gauge_bgcolor, borderwidth=5, bgcolor=st.session_state.gauge_bgcolor,
                    steps=[
                        dict(range=[0, inflation_rate], color="firebrick"),
                        dict(range=[inflation_rate, st.session_state.risk_free_rate], color="darkorange"),
                    ],
                    threshold=dict(
                    line=dict(color="#0077b6", width=5),
                    thickness=0.8,
                    value=msci_world_annual_return,
                    ),
                )
            ))

            fig_portfolio_revenue.update_layout(
                title=dict(text='Portfolio Return past 365 days', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
                margin=dict(l=70, r=70),
                height=400,  # Adjust the height of the chart
                paper_bgcolor=st.session_state.chart_bgcolor,
                plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
                showlegend=True,
                legend=dict(x=0.5, xanchor='center', y=-0.1, orientation='h'),  # Legend at the bottom
                barcornerradius=5,
            )

            fig_portfolio_revenue.add_trace(go.Bar(
                x=[None], y=[None],
                marker=dict(color='firebrick'),
                name='Inflation'
            ))

            fig_portfolio_revenue.add_trace(go.Bar(
                x=[None], y=[None],
                marker=dict(color='cornflowerblue'),
                name='Your Portfolio'
            ))

            fig_portfolio_revenue.add_trace(go.Bar(
                x=[None], y=[None],
                marker=dict(color='darkorange'),
                name='Risk Free Rate'
            ))

            fig_portfolio_revenue.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                marker=dict(color='#0077b6'),
                name='MSCI World Benchmark'
            ))

            fig_portfolio_revenue.update_xaxes(
                visible=False,
            )
            fig_portfolio_revenue.update_yaxes(
                visible=False,
            )

            # Display the chart
            st.plotly_chart(fig_portfolio_revenue, use_container_width=True, config={'displayModeBar': False})

        else:
            range_value = max(abs(portfolio_annual_return), abs(msci_world_data_gauge_number))+2
            
            fig_portfolio_revenue = go.Figure(go.Indicator(
                value=portfolio_annual_return,
                mode="gauge+number", # and not gauge+number+delta"
                number_suffix="%",
                gauge=dict(
                    axis=dict(
                        range=[range_value * (-1), range_value],
                        tickwidth=2, tickcolor="darkgray", ticksuffix="%",
                        tickfont=dict(color=st.session_state.text_color, size=20),
                    ),
                    bar=dict(color='rgba(0,0,0,0)'),
                    bordercolor=st.session_state.gauge_bgcolor, borderwidth=5, bgcolor=st.session_state.gauge_bgcolor,
                    steps=[
                        dict(range=[portfolio_annual_return, 0], color="cornflowerblue", thickness=0.6),
                        dict(range=[0, inflation_rate], color="firebrick"),
                        dict(range=[inflation_rate, st.session_state.risk_free_rate], color="darkorange"),
                    ],
                    threshold=dict(
                    line=dict(color="#0077b6", width=5),
                    thickness=0.8,
                    value=msci_world_data_gauge_number,
                    ),
                )
            ))

            fig_portfolio_revenue.update_layout(
                title=dict(text='Portfolio Return past 365 days', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
                margin=dict(l=70, r=70),
                height=400,  # Adjust the height of the chart
                paper_bgcolor=st.session_state.chart_bgcolor,
                plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
                showlegend=True,
                legend=dict(x=0.5, xanchor='center', y=-0.2, orientation='h'),  # Legend at the bottom
            )

            fig_portfolio_revenue.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='firebrick'),
                showlegend=True,
                name='Rotes Element'
            ))

            fig_portfolio_revenue.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='darkorange'),
                showlegend=True,
                name='Orangenes Element'
            ))

            # Display the chart
            st.plotly_chart(fig_portfolio_revenue, use_container_width=True, config={'displayModeBar': False})       

    with sharpe_loss:
        # Set start date as the purchase date of the stocks
        start_date = datetime.datetime.today() - datetime.timedelta(days=365)
        end_date = datetime.datetime.now().date()

        # Fetch historical data for selected stocks
        data = yf.download(st.session_state.stocks, start=start_date, end=end_date)['Adj Close']
        total_returns_reinvested = pd.DataFrame()

        for stock in st.session_state.stocks:
            hist = yf.Ticker(stock).history(start=start_date, end=end_date)
            hist['Dividends'] = get_dividends(stock)
            total_return_reinvested = total_return_with_reinvestment(hist)
            total_returns_reinvested[stock] = total_return_reinvested

        # Calculate the portfolio's total return by weighting each stock's total return
        weighted_total_return = total_returns_reinvested.multiply(st.session_state.quantities, axis=1)
        portfolio_total_return_reinvested = weighted_total_return.sum(axis=1)
        portfolio_total_return_reinvested_harmonized = portfolio_total_return_reinvested / portfolio_total_return_reinvested.iloc[0] * 100

        # Calculate daily returns
        returns_daily = data.pct_change()

        # Calculate mean daily returns and covariance matrix
        returns_annual = returns_daily.mean() * 250
        cov_annual = returns_daily.cov() * 250

        # Calculate Fixed Weight Portfolio performance
        fixed_weight_data = data.dot(st.session_state.initial_weights)
        fixed_weight_returns_daily = fixed_weight_data.pct_change()
        fixed_weight_return_annual = fixed_weight_returns_daily.mean() * 250
        fixed_weight_volatility_annual = fixed_weight_returns_daily.std() * np.sqrt(250)

        # Calculate Sharpe Ratio
        risk_free_rate_annual = st.session_state.risk_free_rate / 100  # Assuming risk_free_rate is in percentage
        sharpe_ratio = (fixed_weight_return_annual - risk_free_rate_annual) / fixed_weight_volatility_annual

        # Define Sharpe Ratio threshold
        sharpe_ratio_threshold = portfolio_sharpe_ratio if sharpe_ratio > -1 else -0.99

        # Create the gauge chart for Sharpe Ratio
        fig_sharpe_loss = go.Figure(go.Indicator(
            value=portfolio_sharpe_ratio,
            mode="gauge+number",
            gauge=dict(
                axis=dict(
                    range=[-1, 3],
                    tickwidth=2, tickcolor="darkgray",
                    tickfont=dict(color=st.session_state.text_color, size=20),
                    tickmode="array", tickvals=(-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3),
                ),
                bar=dict(color="rgba(0,0,0,0)"),
                bordercolor=st.session_state.gauge_bgcolor, borderwidth=5,
                steps=[
                    dict(range=[-1, -0.5], color="firebrick"),
                    dict(range=[-0.5, 0], color="indianred"),
                    dict(range=[0, 0.5], color="mistyrose"),  # aed581
                    dict(range=[0.5, 1], color="#f1f8e9"),  # aed581
                    dict(range=[1, 1.5], color="#c5e1a5"),
                    dict(range=[1.5, 2], color="#9ccc65"),
                    dict(range=[2, 2.5], color="#7cb342"),
                    dict(range=[2.5, 3], color="#558b2f"),
                ],
                threshold=dict(
                    line=dict(color="black", width=3),
                    thickness=1,
                    value=sharpe_ratio_threshold,
                ),
            )
        ))

        fig_sharpe_loss.update_layout(
            title=dict(text='Portfolio Sharpe Ratio', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
            margin=dict(l=70, r=70),
            height=400,
            paper_bgcolor=st.session_state.chart_bgcolor,
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(x=0.5, xanchor='center', y=-0.2, orientation='h')  # Legend at the bottom
        )

        # Display the chart
        st.plotly_chart(fig_sharpe_loss, use_container_width=True, config={'displayModeBar': False})

    with portfolio_consumption:

        fig_portfolio_consumption = make_subplots(
            rows=1, cols=3, 
            specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]], 
            subplot_titles=("Your Portfolio", "Min. Volatility <br> with same Return", "Max. Return <br> with same Volatilit"),
        )

        fig_portfolio_consumption.add_trace(go.Pie(
            values=st.session_state.weights,
            labels=st.session_state.stocks,
            marker_colors=['#1565c0', '#1976d2', '#1e88e5', '#2196f3', '#42a5f5', '#64b5f6']),
            row=1, col=1,
        )
        
        # Add traces for the Pie Charts with appropriate values and labels
        fig_portfolio_consumption.add_trace(go.Pie(
            values=st.session_state.weights_minvar,
            labels=st.session_state.stocks,
            marker_colors=['#469d89', '#56ab91', '#67b99a', '#78c6a3', '#88d4ab', '#99e2b4']),
            row=1, col=2,
        )

        fig_portfolio_consumption.add_trace(go.Pie(
            values=st.session_state.weights_highret,
            labels=st.session_state.stocks,
            marker_colors=['#f4762d', '#f6863d', '#f8964c', '#faa75c', '#fbb76b', '#fdc77b']),
            row=1, col=3,
        )


        fig_portfolio_consumption.update_traces(
            sort=False,
            textinfo="label+percent",
            direction="clockwise",
            textposition="inside",
            insidetextfont=dict(size=17),
            insidetextorientation="horizontal",
        )

        annotations = []
        for i, title in enumerate(fig_portfolio_consumption['layout']['annotations']):
            title['y'] += 0.1  # Adjust this value to add more space

            # Append modified title to annotations list
            annotations.append(title)

        fig_portfolio_consumption.update_layout(
            annotations=annotations,
            title=dict(text='Portfolio Consumption', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
            margin=dict(t=120, l=40, r=40, b=40),
            height=400,
            paper_bgcolor=st.session_state.chart_bgcolor,
            showlegend=False,
            legend=dict(x=0.5, xanchor='center', y=-0.1, orientation='h'),  # Legend at the bottom
            grid_xgap=1,
        )

        fig_portfolio_consumption.update_annotations(
            font=dict(size=20),
        )
        
        st.plotly_chart(fig_portfolio_consumption, use_container_width=True, config={'displayModeBar': False})

with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    portfolio_performance, monte_carlo = st.columns([1, 1], gap="medium")

    with portfolio_performance:

        # Create a new figure
        fig_performance = go.Figure()

        # Add MSCI World Index trace
        fig_performance.add_trace(go.Scatter(
            x=msci_world_data_harmonized.index,
            y=msci_world_data_harmonized,
            mode='lines',
            name='iShares MSCI World',
            line=dict(color='#0077b6'),
        ))

        # Add iShares MSCI Emerging Markets trace
        fig_performance.add_trace(go.Scatter(
            x=iShares_eem_data_harmonized.index,
            y=iShares_eem_data_harmonized,
            mode='lines',
            name='iShares MSCI Emerging Markets',
            line=dict(color='red'),
            visible="legendonly",
        ))

        # Add iShares US Tech trace
        fig_performance.add_trace(go.Scatter(
            x=iShares_tech_data_harmonized.index,
            y=iShares_tech_data_harmonized,
            mode='lines',
            name='iShares U.S. Tech',
            line=dict(color='green'),
            visible="legendonly",
        ))

        # Add Fixed Weight Portfolio trace
        #fig_performance.add_trace(go.Scatter(
        #    x=portfolio_data_harmonized.index,
        #    y=portfolio_data_harmonized,
        #    mode='lines',
        #    name='Your Portfolio',
        #    line=dict(color='cornflowerblue'),
        #    line_width=3,
        #    legendrank=1,
        #))
        
        # Add Fixed Weight Portfolio trace with total return data
        fig_performance.add_trace(go.Scatter(
            x=portfolio_total_return_reinvested_harmonized.index,
            y=portfolio_total_return_reinvested_harmonized,
            mode='lines',
            name='Your Portfolio',
            line=dict(color='pink'),
            line_width=3,
            legendrank=1,
        ))


        # Update layout
        fig_performance.update_layout(
            title=dict(text='Performance: Portfolio vs. Benchmark (1 year)', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
            yaxis_title='Harmonized Value',
            margin=dict(l=100, r=30, t=60, b=40),
            height=400,
            paper_bgcolor=st.session_state.chart_bgcolor,
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.45, xanchor='center', y=-0.1, orientation='h', itemwidth=30, indentation=30),
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[
                                {
                                    'x': [msci_world_data_harmonized.index, iShares_eem_data.index, iShares_tech_data_harmonized.index, portfolio_data_harmonized.index],
                                    'y': [msci_world_data_harmonized, iShares_eem_data_harmonized, iShares_tech_data_harmonized, portfolio_data_harmonized]
                                }
                            ],
                            label='Absolute Values',
                            method='update'
                        ),
                        dict(
                            args=[
                               {
                                    'x': [msci_world_data_harmonized.index, iShares_eem_data.index, iShares_tech_data_harmonized.index, portfolio_data_harmonized.index],
                                    'y': [msci_world_data_sharpe, iShares_eem_data_sharpe, iShares_tech_data_sharpe, portfolio_data_sharpe]
                                }
                            ],
                            label='Risk Adjusted',
                            method='update'
                        ),
                    ],
                    font=dict(color='cornflowerblue', size=15),  # weight removed
                    bordercolor='cornflowerblue',
                    borderwidth=2,
                    direction='left',
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    x=1.03,
                    xanchor='right',
                    y=1.15,
                    yanchor='top',
                    type="buttons",
                )
            ]
        )

        # Display the chart
        st.plotly_chart(fig_performance, use_container_width=True, config={'displayModeBar': False})

    with monte_carlo:
        # Create a Plotly figure
        fig_monte_carlo = go.Figure()

        fig_monte_carlo = go.Figure()

        fig_monte_carlo.add_trace(go.Scatter(
            x=st.session_state.mc_allvol, y=st.session_state.mc_allret,
            mode='markers',
            marker=dict(
                size=10,  
                color=st.session_state.mc_all_sharpe, # Color by Sharpe Ratio
                colorscale='Pinkyl',  # Color scale to use
                showscale=True,  # Show color scale
                line=dict(width=0.3, color=st.session_state.gauge_bgcolor),
                colorbar=dict(borderwidth=0, outlinecolor= st.session_state.gauge_bgcolor, outlinewidth=4,),
            ),
            text="Halleluja",  # Tooltip text
            name='Portfolios',
            opacity=1,
            showlegend=False,
            #hoverinfo='skip',
        ))

        # Efficient Frontier
        fig_monte_carlo.add_trace(go.Scatter(
            x=st.session_state.mc_effvol, y=st.session_state.mc_effret,
            mode='lines',
            line=dict(dash='dash', color=st.session_state.text_color, width=2),
            name='Efficient Frontier',
            showlegend=False,
            hoverinfo='skip',
        ))

        # Your Portfolio
        fig_monte_carlo.add_trace(go.Scatter(
            x=[st.session_state.mc_fixvol], y=[st.session_state.mc_fixret],
            mode='markers',
            marker=dict(
                color='#1565c0', # blue
                size=20,
                line_width=5,
                symbol='x-thin-open'
            ),
            name='Your Portfolio',
            legendrank=1,
        ))

        # Lowest Volatility
        fig_monte_carlo.add_trace(go.Scatter(
            x=[st.session_state.mc_lowvol], y=[st.session_state.mc_lowret],
            mode='markers',
            marker=dict(
                color='#469d89', # green
                size=20,
                line_width=5,
                symbol='x-thin-open'
            ),
            name='Min. Volatility with same Return',
            legendrank=2,
        ))

        # Highest Return
#        if st.session_state.mc_higretsamvol:
        fig_monte_carlo.add_trace(go.Scatter(
            x=[st.session_state.mc_highvol], y=[st.session_state.mc_highret],
            mode='markers',
            marker=dict(
                color='#f4762d', # orange
                size=20,
                line_width=5,
                symbol='x-thin-open'
            ),
            name='Max. Return with same Volatility',
            legendrank=3,
        ))

        # Minimum Variance Portfolio
        fig_monte_carlo.add_trace(go.Scatter(
            x=[st.session_state.mc_minvol], y=[st.session_state.mc_minret],
            mode='markers',
            marker=dict(
                color=st.session_state.text_color,
                size=20,
                line_width=5,
                symbol='x-thin-open'
            ),
            name='Minimum Variance Portfolio',
            legendrank=4,
        ))

        # Update layout
        fig_monte_carlo.update_layout(
            xaxis_title='Volatility (Std. Deviation)',
            yaxis_title='Expected Returns',
            title=dict(text='Efficient Frontier', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
            margin=dict(l=100, r=30, t=60, b=40),
            height=400,  # Adjust the height of the chart
            paper_bgcolor=st.session_state.chart_bgcolor,  # Background color
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.5, xanchor='center', y=-0.22, orientation='h'),  # Legend at the bottom
        )

        fig_monte_carlo.update_yaxes(
            tickformat='.2%',
            range=[min(st.session_state.mc_minret, st.session_state.mc_fixret)*0.85, st.session_state.mc_highret*1.05],
        )

        fig_monte_carlo.update_xaxes(
            tickformat='.2%',
            range=[st.session_state.mc_minvol*0.995, st.session_state.mc_fixvol*1.005],
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig_monte_carlo)#, config={'displayModeBar': False})