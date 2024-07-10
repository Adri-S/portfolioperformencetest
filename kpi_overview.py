import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
import pandas_datareader.data as web
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def display_kpi_overview():
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
    msci_world_data = yf.download('URTH', start=start_date, end=end_date)['Adj Close']
    msci_world_data = msci_world_data / msci_world_data.iloc[0] * 100 # Normalize the data

    # Fetch DOW Jones Index data
    dow_world_data = yf.download('^DJI', start=start_date, end=end_date)['Adj Close']
    dow_world_data = dow_world_data / dow_world_data.iloc[0] * 100  # Normalize the data

    # Fetch NASDAQ 100 Index data
    nasdaq_world_data = yf.download('^NDX', start=start_date, end=end_date)['Adj Close']
    nasdaq_world_data = nasdaq_world_data / nasdaq_world_data.iloc[0] * 100  # Normalize the data

    # Calculate Fixed Weight Portfolio performance
    stock_price_data = yf.download(st.session_state.stocks, start=start_date, end=end_date)['Adj Close']
    stock_price_data = stock_price_data[st.session_state.stocks]
    stock_volume_data = stock_price_data * st.session_state.quantities
    total_volume = stock_volume_data.sum(axis=1)
    total_volume_harmonized = total_volume / total_volume.iloc[0] * 100  # Normalize the data
    portfolio_return_gc = (total_volume.iloc[-1] - total_volume.iloc[0]) / total_volume.iloc[0] * 100


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

    # risk free rate on base of the 13 week US treasury bill
    start_date_cpi = datetime.datetime.today() - datetime.timedelta(days=365)
    treasury_data = yf.download('^IRX', start=start_date_cpi)
    risk_free_rate = treasury_data.iloc[0,3]

    # Fetch MSCI World Index data
    msci_world_data_gauge = yf.download('URTH', start=datetime.datetime.today()- datetime.timedelta(days=365), end=datetime.datetime.now().date())['Adj Close']
    last = msci_world_data_gauge.shape[0] - 1  # Index of the last row
    msci_world_data_gauge_number = (msci_world_data_gauge.iloc[0] - msci_world_data_gauge.iloc[0]) / msci_world_data_gauge.iloc[0] * 100

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

            if portfolio_return_gc >= 0:
                fig_portfolio_revenue = go.Figure(go.Indicator(
                    value=portfolio_return_gc,
                    mode="gauge+number", # and not gauge+number+delta"
                    number_suffix="%",
                    gauge=dict(
                        axis=dict(
                            range=[0, max(portfolio_return_gc, inflation_rate, risk_free_rate, msci_world_data_gauge_number)+1],
                            tickwidth=2, tickcolor="darkgray", ticksuffix="%",
                            tickfont=dict(color=st.session_state.text_color, size=20),
                        ),
                        bar=dict(color="cornflowerblue"),
                        bordercolor=st.session_state.gauge_bgcolor, borderwidth=5, bgcolor=st.session_state.gauge_bgcolor,
                        steps=[
                            dict(range=[0, inflation_rate], color="firebrick"),
                            dict(range=[inflation_rate, risk_free_rate], color="darkorange"),
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

                fig_portfolio_revenue.add_annotation(
                    width=300,
                    text="An annotation whose text and arrowhead reference the axes and the data",
                    x=0.5, xanchor='center', y=-0.2,
                    showarrow=False,
                )

                # Display the chart
                st.plotly_chart(fig_portfolio_revenue, use_container_width=True, config={'displayModeBar': False})

            else:
                range_value = max(abs(portfolio_return_gc), abs(msci_world_data_gauge_number))+2
                
                fig_portfolio_revenue = go.Figure(go.Indicator(
                    value=portfolio_return_gc,
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
                            dict(range=[portfolio_return_gc, 0], color="cornflowerblue", thickness=0.6),
                            dict(range=[0, inflation_rate], color="firebrick"),
                            dict(range=[inflation_rate, risk_free_rate], color="darkorange"),
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

                fig_portfolio_revenue.add_annotation(
                    width=300,
                    text="An annotation whose text and arrowhead reference the axes and the data",
                    x=0.5, xanchor='center', y=-0.2,
                    showarrow=False,
                )

                # Display the chart
                st.plotly_chart(fig_portfolio_revenue, use_container_width=True, config={'displayModeBar': False})       

        with sharpe_loss:
            # Set start date as the purchase date of the stocks
            start_date = datetime.datetime.today()- datetime.timedelta(days=365)
            end_date = datetime.datetime.now().date()

            # Fetch historical data for selected stocks
            data = yf.download(st.session_state.stocks, start=start_date, end=end_date)['Adj Close']

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
            risk_free_rate_annual = risk_free_rate / 100  # Assuming risk_free_rate is in percentage
            sharpe_ratio = (fixed_weight_return_annual - risk_free_rate_annual) / fixed_weight_volatility_annual

            # Define Sharpe Ratio threshold
            sharpe_ratio_threshold = sharpe_ratio if sharpe_ratio > -1 else -0.99

            # Create the gauge chart for Sharpe Ratio
            fig_sharpe_loss = go.Figure(go.Indicator(
                value=sharpe_ratio,
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
                title=dict(text='Sharpe Ratio', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
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
                subplot_titles=("Your Portfolio", "Minimum Variance", "Sharpe"),
            )

            fig_portfolio_consumption.add_trace(go.Pie(
                values=st.session_state.weights,
                labels=st.session_state.stocks,
                marker_colors=['#48578E', '#6171A9', '#7D8EC4', '#9DADDF']),
                row=1, col=1,
            )
            
            # Add traces for the Pie Charts with appropriate values and labels
            fig_portfolio_consumption.add_trace(go.Pie(
                values=st.session_state.weights_minvar,
                labels=st.session_state.stocks,
                marker_colors=['#689638', '#7cb342', '#8bc34a', '#9ccc65']),
                row=1, col=2,
            )

            fig_portfolio_consumption.add_trace(go.Pie(
                values=st.session_state.weights_highret,
                labels=st.session_state.stocks,
                marker_colors=['#FF7B00', '#FF8D21', '#FFA652', '#FFB76B']),
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

            fig_portfolio_consumption.update_layout(
                title=dict(text='Portfolio Consumption', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
                margin=dict(l=40, r=40),
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
                x=msci_world_data.index,
                y=msci_world_data,
                mode='lines',
                name='MSCI World Index',
                line=dict(color='#0077b6'),
            ))

            # Add DOW Jones Index trace
            fig_performance.add_trace(go.Scatter(
                x=dow_world_data.index,
                y=dow_world_data,
                mode='lines',
                name='DOW Jones',
                line=dict(color='yellow'),
                visible="legendonly",
            ))

            # Add NASDAQ 100 Index trace
            fig_performance.add_trace(go.Scatter(
                x=nasdaq_world_data.index,
                y=nasdaq_world_data,
                mode='lines',
                name='NASDAQ 100',
                line=dict(color='red'),
                visible="legendonly",
            ))

            # Add Fixed Weight Portfolio trace
            fig_performance.add_trace(go.Scatter(
                x=total_volume_harmonized.index,
                y=total_volume_harmonized,
                mode='lines',
                name='Fixed Weight Portfolio',
                line=dict(color='cornflowerblue'),
                line_width=3,
                legendrank=1,
            ))

            # Update layout
            fig_performance.update_layout(
                title=dict(text='Portfolio Performance vs Benchmark', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
                xaxis_title='Date',
                yaxis_title='Normalized Value',
                margin=dict(l=100, r=30, t=60, b=40),
                height=400,
                paper_bgcolor=st.session_state.chart_bgcolor,
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend_font_color=st.session_state.text_color,
                legend=dict(x=0.45, xanchor='center', y=-0.2, orientation='h', itemwidth=30, indentation=30),
            )

            # Display the chart
            st.plotly_chart(fig_performance, use_container_width=True, config={'displayModeBar': False})

        with monte_carlo:
            # Create a Plotly figure
            fig_monte_carlo = go.Figure()

            # Scatter plot for portfolios
            fig_monte_carlo.add_trace(go.Scatter(
                x=st.session_state.mc_allvol, y=st.session_state.mc_allret,
                mode='markers',
                marker=dict(
                    size=10,
                    color="green",  # Color by Sharpe Ratio
                    colorscale='RdYlGn',
                    showscale=True,
                    line=dict(width=1, color='black')
                ),
                text="Simulation Points",  # Tooltip text
                name='Portfolios',
                opacity=0.4,
            ))

            # Efficient Frontier
            fig_monte_carlo.add_trace(go.Scatter(
                x=st.session_state.mc_effvol, y=st.session_state.mc_effret,
                mode='lines',
                line=dict(dash='dash', color='black', width=2),
                name='Efficient Frontier'
            ))

            # Fixed Weight Portfolio
            fig_monte_carlo.add_trace(go.Scatter(
                x=[st.session_state.mc_fixvol], y=[st.session_state.mc_fixret],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=15,
                    symbol='star'
                ),
                name='Fixed Weight Portfolio'
            ))

            # Highest Return with Same or Lower Volatility
            if st.session_state.mc_higretsamvol:
                fig_monte_carlo.add_trace(go.Scatter(
                    x=[st.session_state.mc_highvol], y=[st.session_state.mc_highret],
                    mode='markers',
                    marker=dict(
                        color='orange',
                        size=10,
                        symbol='circle'
                    ),
                    name='Highest Return with Same or Lower Volatility'
                ))

            # Lowest Volatility with Same or Higher Return
            fig_monte_carlo.add_trace(go.Scatter(
                x=[st.session_state.mc_lowvol], y=[st.session_state.mc_lowret],
                mode='markers',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='square'
                ),
                name='Lowest Volatility with Same or Higher Return'
            ))

            # Minimum Variance Portfolio
            fig_monte_carlo.add_trace(go.Scatter(
                x=[st.session_state.mc_minvol], y=[st.session_state.mc_minret],
                mode='markers',
                marker=dict(
                    color='black',
                    size=10,
                    symbol='diamond'
                ),
                name='Minimum Variance Portfolio'
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
                legend=dict(x=0.45, xanchor='center', y=-0.1, orientation='h'),  # Legend at the bottom
            )

            fig_monte_carlo.update_yaxes(
                maxallowed=st.session_state.mc_highret+0.02,
                minallowed=min(st.session_state.mc_minret, st.session_state.mc_fixret)-0.02, # minallowed=min(max(st.session_state.mc_minret, st.session_state.mc_lowret), st.session_state.mc_fixret)-0.02,
            )

            fig_monte_carlo.update_xaxes(
                maxallowed=max(st.session_state.mc_highvol, st.session_state.mc_fixvol)+0.002,
                minallowed=st.session_state.mc_minvol-0.0001,
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig_monte_carlo)