import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly
import plotly.graph_objects as go
import datetime

if 'background_color_page' not in st.session_state:
    st.session_state.background_color_page="snow"
    st.session_state.chart_bgcolor="#ffffff"
    st.session_state.text_color="black"

# Setting up the page configuration
st.set_page_config(
    page_title="Interactive Sidebar",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for general styling
st.markdown(f"""
    <!-- Bootstrap Icons CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">

    <style>
        .container1 {{
            padding: 15px;
        }}
        header[data-testid="stHeader"] {{
            background-color: {st.session_state.background_color_page};
            color: {st.session_state.text_color};
        }}
        .st-cr {{
            border-color: {st.session_state.border_color_input};
        }}   
        div.stTextInput input,
        div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] {{
            background-color: {st.session_state.chart_bgcolor};
            color: {st.session_state.text_color};
        }}
        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="InputInstructions"] {{
            color: {st.session_state.text_color};
        }}
        .stPlotlyChart {{
            border-radius: 15px;
            overflow: hidden;
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {{
            background-color: {st.session_state.background_color_page};
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
            top: 10px;
            left: 30px;
        }}
        .subheading-stocks {{
            position: absolute;
            font-weight: normal;
            font-size: 25px;
            color: {st.session_state.text_color};
            top: 10px;
            right: 10px;
        }}
        .stocks-top-numbers {{
            position: absolute;
            font-weight: bold;
            font-size: 30px;
            color: {st.session_state.text_color};
            bottom: 10px;
            right: 10px;
        }}
        .custom-table {{
            width: 100%;
            font-size: 18px;
            border: none;
        }}
        .custom-table thead {{
            background-color: {st.session_state.chart_bgcolor};
            font-size: 25px;
            color: cornflowerblue;
        }}
        .custom-table thead th:first-child {{
            border-top-left-radius: 15px;
            border-color: rgba(0,0,0,0);
            text-align: left;
        }}
        .custom-table thead th:last-child {{
            border-top-right-radius: 15px;
            border-color: rgba(0,0,0,0);
            text-align: right;
        }}
        .custom-table tbody {{
            background-color: {st.session_state.chart_bgcolor};
            color: {st.session_state.text_color};
            border-color: rgba(0,0,0,0);
        }}
        .custom-table tbody td:first-child {{
            border-color: rgba(0,0,0,0);
            text-align: left;
        }}
        .custom-table tbody td:last-child {{
            border-color: rgba(0,0,0,0);
            text-align: right;
        }}
        .custom-table tr:last-child td:first-child {{
            border-bottom-left-radius: 15px;
            font-weight: bold;
            padding-top: 2px;
            padding-bottom: 2px;
        }}
        .custom-table tr:last-child td:last-child {{
            border-bottom-right-radius: 15px;
            font-weight: bold;
            padding-top: 2px;
            padding-bottom: 2px;
        }}
        .custom-table tr td {{
            padding-top: 0;
            padding-bottom: 0;
        }}
    </style>
""", unsafe_allow_html=True)

# Function to fetch stock data and prepare financial statements
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)

    # Get stock price and market cap
    stock_info = stock.info
    share_price = stock_info.get('currentPrice', 'N/A')
    market_cap = stock_info.get('marketCap', 'N/A')
    fifty_day_average = stock_info.get('fiftyDayAverage', 'N/A')

    # Get financial statements
    income_statement = stock.financials.T
    balance_sheet = stock.balance_sheet.T

    # Convert dates to strings for dropdown
    income_statement['Year'] = income_statement.index.year.astype(str)
    balance_sheet['Year'] = balance_sheet.index.year.astype(str)

    return stock_info, share_price, market_cap, fifty_day_average, income_statement, balance_sheet

# Title
st.title("Welcome to Stock Explorer")

# Container 1 with 4 columns
with st.container():
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    stocknumberformat = "{:0,.2f}"
    
    with col1:

        ticker_input = st.text_input("Enter Stock Ticker Here:", value='AAPL', max_chars=6)

        if ticker_input:
            # Fetch stock data
            stock_info, share_price, market_cap, fifty_day_average, income_statement, balance_sheet = get_stock_data(ticker_input)
            if share_price == "N/A":
                st.warning(f"Could not find data for stock ticker '{ticker_input}'. Please enter a valid ticker.")
                st.stop()  # Stop further execution
            
    with col2:
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: #ffba08; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="bi bi-cash-coin" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 26px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">Share Price:</div>
                <div class="corner-text stocks-top-numbers">{stocknumberformat.format(share_price)} $</div>
            </div>
        """, unsafe_allow_html=True)
  
    with col3:
        if fifty_day_average <= share_price:
            fda_bgc = "#62BD7D"
            fda_icon = "bi bi-graph-up-arrow"
        else:
            fda_bgc = "#C84455"
            fda_icon = "bi bi-graph-down-arrow"
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: {fda_bgc}; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="{fda_icon}" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">Fifty Day Average:</div>
                <div class="corner-text stocks-top-numbers">{stocknumberformat.format(fifty_day_average)} $</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="column-box-stocks-1">
                <span class="bi bi-square-fill" style="color: #0077b6; font-size: 4.5rem; position: absolute; top: -5px; left: 15px; z-index: 1; padding: 0px;"></span>
                <span class="bi bi-opencollective" style="color: #ffffff; font-size: 2.5rem; position: absolute; top: 20px; left: 31px; z-index: 1; padding: 0px;"></span>
                <div class="corner-text subheading-stocks">Market Cap:</div>
                <div class="corner-text stocks-top-numbers">{stocknumberformat.format(market_cap/1000000000)} B$</div>
            </div>
        """, unsafe_allow_html=True)

# Container 2 with 3 columns

with st.container():
    st.markdown('<div class="container1">', unsafe_allow_html=True)
    
    income_statement_table, net_income_chart, balance_sheet_chart = st.columns([1, 1, 2], gap="medium")
    
    with income_statement_table:
        
        stock_info, share_price, market_cap, fifty_day_average, income_statement, balance_sheet = get_stock_data(ticker_input)
        years = income_statement['Year'].unique()
        selected_year = st.selectbox("Select Year Here:", years)

        # Filter data for the selected year
        income_statement_year = income_statement[income_statement['Year'] == selected_year].drop(columns=['Year'])

        # From a list of dictionaries
        income_statement_data = [
            {'Income Statement': 'Revenue', selected_year: income_statement_year["Total Revenue"].values[0]},
            {'Income Statement': '(-) Cost Of Revenue', selected_year: income_statement_year["Cost Of Revenue"].values[0]},
            {'Income Statement': '= Gross Profit', selected_year: income_statement_year["Gross Profit"].values[0]},
            {'Income Statement': '(-) Operating Expense', selected_year: income_statement_year["Operating Expense"].values[0]},
            {'Income Statement': '= Operating Income', selected_year: income_statement_year["Operating Income"].values[0]},
            {'Income Statement': '(+-) Other Income/Expense', selected_year: income_statement_year["Pretax Income"].values[0] - income_statement_year["Operating Income"].values[0]},
            {'Income Statement': '= Income Before Tax', selected_year: income_statement_year["Pretax Income"].values[0]},
            {'Income Statement': '(-) Tax Provision', selected_year: income_statement_year["Tax Provision"].values[0]},
            {'Income Statement': '= Net Income', selected_year: income_statement_year["Net Income Common Stockholders"].values[0]},
        ]

        def format_amount(value):
            if abs(value) >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f} B$"
            elif abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.2f} M$"
            else:
                return f"{value} $"

        # Create the DataFrame
        income_statement_table = pd.DataFrame(income_statement_data)
        income_statement_table[selected_year] = income_statement_table[selected_year].apply(format_amount)

        # Convert custom data to HTML and apply custom table class
        custom_table_html = income_statement_table.to_html(classes='custom-table', index=False)
        st.markdown(custom_table_html, unsafe_allow_html=True)

    with net_income_chart:          
        # Ensure 'Year' is numeric, if not already
        income_statement['Year'] = pd.to_numeric(income_statement['Year'], errors='coerce')

        # Filter the DataFrame to include only years from 2020 to 2023
        filtered_data = income_statement[(income_statement['Year'] >= 2020) & (income_statement['Year'] <= 2023)]

        # Create bar chart for Total Revenue and Net Income
        fig_income_chart = go.Figure()

        # Net Profit Margin Ratio
        filtered_data['Net Income Common Stockholders'] = pd.to_numeric(filtered_data['Net Income Common Stockholders'], errors='coerce')
        filtered_data['Total Revenue'] = pd.to_numeric(filtered_data['Total Revenue'], errors='coerce')
        filtered_data.dropna(subset=['Net Income Common Stockholders', 'Total Revenue'], inplace=True)
        result = filtered_data['Net Income Common Stockholders'] / filtered_data['Total Revenue'] * 100
        net_profit_margin_ratio = result.round(2)

        # Add bars for Total Revenue
        fig_income_chart.add_trace(go.Bar(
            name='Total Revenue',
            x=filtered_data['Year'],
            y=filtered_data['Total Revenue'],
            marker=dict(color='#62BD7D', opacity=1),
        ))

        # Add bars for Net Income
        fig_income_chart.add_trace(go.Bar(
            name='Net Income',
            x=filtered_data['Year'],
            y=filtered_data['Net Income Common Stockholders'],
            marker=dict(color='#C84455'),
        ))

        # Add bars for Net Income
        fig_income_chart.add_trace(go.Bar(
            name='Net Profit Margin',
            x=filtered_data['Year'],
            y=[0, 0, 0, 0, 0, 0],
            marker=dict(color='cornflowerblue'),
        ))

        # Add annotations for Net Profit Margin with a black background
        annotations = []
        for idx, (year, net_income) in enumerate(zip(filtered_data['Year'], filtered_data['Net Income Common Stockholders'])):
            annotations.append(dict(
                x=year,
                y=net_income + (filtered_data['Total Revenue'].max() * 0.08),  # Adjust y position as needed
                text=f"{net_profit_margin_ratio.iloc[idx]}%",
                showarrow=False,
                font=dict(size=17, color='white'),
                align='center',
                bordercolor='#0077b6',
                borderwidth=1,
                borderpad=3,
                bgcolor='cornflowerblue',
                opacity=1,
            ))

        # Update layout to match desired style
        fig_income_chart.update_layout(
            title=dict(text='Net Profit Margin', y=1, x=0, pad_l=20, pad_t=20, font=dict(color='cornflowerblue', size=35)),
            barmode='overlay', # or 'group'
            bargroupgap=0.2,
            barcornerradius=5,
            xaxis=dict(tickmode='linear'),  # Ensure each year is displayed
            yaxis=dict(title='Amount',),
            margin=dict(l=80, r=30, t=70, b=40),
            height=400,  # Adjust the height of the chart
            paper_bgcolor=st.session_state.chart_bgcolor,  # Background color
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.5, xanchor='center', y=-0.1, orientation='h'), # Legend at the bottom
            annotations=annotations  # Add annotations to the layout
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig_income_chart, use_container_width=True, config={'displayModeBar': False})

    with balance_sheet_chart:
        
        # Function to fetch balance sheet data from Yahoo Finance using yfinance
        def get_balance_sheet_data(ticker):
            # Fetch the financials using yfinance
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet.T
            
            # Check if required columns exist
            required_columns_balance = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
            missing_columns_balance = [col for col in required_columns_balance if col not in balance_sheet.columns]
            
            if missing_columns_balance:
                st.warning(f"Missing columns in balance sheet data: {', '.join(missing_columns_balance)}")
                return pd.DataFrame()  # Return an empty DataFrame if required columns are missing
            
            # Prepare the DataFrame
            balance_sheet.reset_index(inplace=True)
            balance_sheet.rename(columns={'index': 'Year'}, inplace=True)
            
            # Convert 'Year' to datetime and keep only the last 4 years
            balance_sheet['Year'] = pd.to_datetime(balance_sheet['Year'], errors='coerce').dt.year
            last_five_years = balance_sheet.sort_values(by='Year').tail(4)
            
            return last_five_years[['Year', 'Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']]

        # Fetch balance sheet data
        balance_sheet_data = get_balance_sheet_data(ticker_input)

        # Create balance sheet bar chart
        fig_balance_sheet = go.Figure()
        
        # Add bars for 'Total Assets'
        fig_balance_sheet.add_trace(go.Bar(
            name='Assets',
            x=balance_sheet_data['Year'],
            y=balance_sheet_data['Total Assets'],
            marker=dict(color='#62BD7D')  # Customize bar color
        ))
        
        # Add bars for 'Total Liabilities Net Minority Interest'
        fig_balance_sheet.add_trace(go.Bar(
            name='Liabilities',
            x=balance_sheet_data['Year'],
            y=balance_sheet_data['Total Liabilities Net Minority Interest'],
            marker=dict(color='#C84455')  # Customize bar color
        ))
    
        # Add line for 'Stockholders Equity'
        fig_balance_sheet.add_trace(go.Scatter(
            name='Equity',
            x=balance_sheet_data['Year'],
            y=balance_sheet_data['Stockholders Equity'],
            mode='lines+markers',
            line=dict(color='orange', width=2),
            marker=dict(symbol='circle', size=10)
        ))

        # Update layout to match desired style
        fig_balance_sheet.update_layout(
            title=dict(text='Balance Sheet', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
            barmode='group',
            bargroupgap=0.2,
            barcornerradius=5,
            yaxis_title='Amount',
            xaxis=dict(tickmode='linear'),  # Ensure each year is displayed
            margin=dict(l=80, r=30, t=60, b=40),
            height=400,  # Adjust the height of the chart
            paper_bgcolor=st.session_state.chart_bgcolor,  # Background color
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.45, xanchor='center', y=-0.1, orientation='h')  # Legend at the bottom
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig_balance_sheet, use_container_width=True, config={'displayModeBar': False})

# Container 3 with 2 columns
with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    market_value, cash_flow_chart = st.columns([1, 1], gap="medium")
    
    with market_value:

        # Get today's date
        end_date = datetime.datetime.now().date()
        
        # Define periods for data fetching
        periods = {
            'Last 5 Years': (end_date - datetime.timedelta(days=5*365), end_date),
            'Last Year': (end_date - datetime.timedelta(days=365), end_date),
            'Last Quarter': (end_date - datetime.timedelta(days=90), end_date),
            'Year to Date': (datetime.date(datetime.datetime.now().year, 1, 1), end_date),
        }
        initial_period = 'Last 5 Years'
        
        # Fetch data for all periods
        market_data = {}
        for period, (start_date, end_date) in periods.items():
            data = yf.download(ticker_input, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            market_data[period] = data
        
        stock = yf.Ticker(ticker_input)
        hist_5y = stock.history(period="5y")
        hist_5y.reset_index(inplace=True)
        hist_5y['Dividends'] = hist_5y['Dividends'].fillna(0)
        adj_dividends_5y = hist_5y['Dividends'].cumsum()
        dividends_5y = market_data['Last 5 Years']['Close'] + adj_dividends_5y

        hist_1y = stock.history(period="1y")
        hist_1y.reset_index(inplace=True)
        hist_1y['Dividends'] = hist_1y['Dividends'].fillna(0)
        adj_dividends_1y = hist_1y['Dividends'].cumsum()
        dividends_1y = market_data['Last Year']['Close'] + adj_dividends_1y

        hist_1q = stock.history(period="3mo")
        hist_1q.reset_index(inplace=True)
        hist_1q['Dividends'] = hist_1q['Dividends'].fillna(0)
        adj_dividends_1q = hist_1q['Dividends'].cumsum()
        dividends_1q = market_data['Last Quarter']['Close'] + adj_dividends_1q

        hist_ytd = stock.history(period="YTD")
        hist_ytd.reset_index(inplace=True)
        hist_ytd['Dividends'] = hist_ytd['Dividends'].fillna(0)
        adj_dividends_ytd = hist_ytd['Dividends'].cumsum()
        dividends_ytd = market_data['Year to Date']['Close'] + adj_dividends_ytd
        
        # Create Plotly figure
        fig_market_performance = go.Figure()

        fig_market_performance.add_trace(go.Scatter(
            name=f'{ticker_input} with reinvested dividends',
            line=dict(color='#62BD7D', width=2.5, simplify=True),
            marker_color='#62BD7D',
            x=market_data[initial_period]['Date'],
            y=dividends_5y,
            mode='lines',
            legendrank=2,
            visible="legendonly",
        ))
        
        # Add a trace for the initial view (e.g., Last Year)

        fig_market_performance.add_trace(go.Scatter(
            name=ticker_input,
            line=dict(color='orange', width=2.5, simplify=True),
            marker_color='orange',
            x=market_data[initial_period]['Date'],
            y=market_data[initial_period]['Close'],
            mode='lines',
            legendrank=1,
        ))

        # Update layout with updatemenus for period selection
        fig_market_performance.update_layout(
            title=dict(text='Stock Price Over Time', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
            yaxis_title='Close Price',
            margin=dict(l=100, r=30, t=60, b=40),
            height=400,
            paper_bgcolor=st.session_state.chart_bgcolor,
            plot_bgcolor='rgba(0,0,0,0)',
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{
                                'x': [market_data['Last 5 Years']['Date']],
                                'y': [dividends_5y, market_data['Last 5 Years']['Close']]
                            }],
                            label='Last 5 years',
                            method='update'
                        ),
                        dict(
                            args=[{
                                'x': [market_data['Last Year']['Date']],
                                'y': [dividends_1y, market_data['Last Year']['Close']]
                            }],
                            label='Last Year',
                            method='update',
                        ),
                        dict(
                            args=[{
                                'x': [market_data['Last Quarter']['Date']],
                                'y': [dividends_1q, market_data['Last Quarter']['Close']]
                            }],
                            label='Last Quarter',
                            method='update'
                        ),
                        dict(
                            args=[{
                                'x': [market_data['Year to Date']['Date']],
                                'y': [dividends_ytd, market_data['Year to Date']['Close']]
                            }],
                            label='YTD',
                            method='update'
                        ),
                    ],
                    font=dict(color='cornflowerblue', size=15),  # weight removed
                    bordercolor='cornflowerblue',
                    borderwidth=2,
                    direction='down',
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    x=1.03,
                    xanchor='right',
                    y=1.15,
                    yanchor='top',
#                    type="buttons",
                )
            ],
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.45, xanchor='center', y=-0.1, orientation='h', itemwidth=30, indentation=60),
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig_market_performance, use_container_width=True, config={'displayModeBar': False})

    with cash_flow_chart:

        # Function to fetch cash flow data from Yahoo Finance using yfinance
        def get_cash_flow_data(ticker):
            # Fetch the financials using yfinance
            stock = yf.Ticker(ticker)
            cashflow = stock.cashflow.T

            # Known column names from yfinance
            known_columns = {
                'Operating Cash Flow': 'Operating Cash Flow',
                'Investing Cash Flow': 'Investing Cash Flow',
                'Financing Cash Flow': 'Financing Cash Flow',
                'Free Cash Flow': 'Free Cash Flow'
            }

            # Check if required columns exist
            missing_columns = [col for col in known_columns.keys() if col not in cashflow.columns]

            if missing_columns:
                st.warning(f"Missing columns in cash flow data: {', '.join(missing_columns)}")
                return pd.DataFrame()  # Return an empty DataFrame if required columns are missing

            # Prepare the DataFrame
            cashflow.reset_index(inplace=True)
            cashflow.rename(columns={'index': 'fiscalDateEnding'}, inplace=True)

            # Convert 'fiscalDateEnding' to datetime and keep only the last four years
            cashflow['fiscalDateEnding'] = pd.to_datetime(cashflow['fiscalDateEnding'], errors='coerce').dt.year
            last_four_years = cashflow.sort_values(by='fiscalDateEnding').tail(4)

            return last_four_years[['fiscalDateEnding'] + list(known_columns.values())]

        # Fetch cash flow data
        cash_flow_data = get_cash_flow_data(ticker_input)

        # Define periods for plotting
        end_date = datetime.datetime.now().date()
        periods = {
            'Last 4 Years': (end_date - datetime.timedelta(days=4 * 365), end_date),
        }

        # Check if data was fetched successfully
        if cash_flow_data.empty:
            st.error(f"No data available for ticker {ticker_input}.")
        else:
            # Create a dictionary to hold data for different periods
            cash_flow_periods = {}
            for period_name, (start_date, end_date) in periods.items():
                filtered_data = cash_flow_data[
                    (cash_flow_data['fiscalDateEnding'] >= pd.to_datetime(start_date).year) &
                    (cash_flow_data['fiscalDateEnding'] <= pd.to_datetime(end_date).year)
                ]
                cash_flow_periods[period_name] = filtered_data

            # Create Plotly figure
            fig_cash_flow = go.Figure()

            # Add initial trace for the 'Last 4 Years'
            initial_period = 'Last 4 Years'
            cash_flow_data = cash_flow_periods[initial_period]

            # Plot each component of the cash flow as bar charts, except for Free Cash Flow as a line chart
            components = {
                'Operating': 'Operating Cash Flow',
                'Investing': 'Investing Cash Flow',
                'Financing': 'Financing Cash Flow'
            }

            # Add bars for 'Operating Cash Flow'
            fig_cash_flow.add_trace(go.Bar(
                name='Operating',
                x=cash_flow_data['fiscalDateEnding'],
                y=cash_flow_data['Operating Cash Flow'],
                marker=dict(color='#62BD7D')  # Customize bar color
            ))

            # Add bars for 'Investing Cash Flow'
            fig_cash_flow.add_trace(go.Bar(
                name='Investing',
                x=cash_flow_data['fiscalDateEnding'],
                y=cash_flow_data['Investing Cash Flow'],
                marker=dict(color='skyblue')  # Customize bar color
            ))

            # Add bars for 'Financing Cash Flow'
            fig_cash_flow.add_trace(go.Bar(
                name='Financing',
                x=cash_flow_data['fiscalDateEnding'],
                y=cash_flow_data['Financing Cash Flow'],
                marker=dict(color='#C84455')  # Customize bar color
            ))

            # Add bars for 'Free Cash Flow'
            fig_cash_flow.add_trace(go.Scatter(
                name='Free Cash Flow',
                x=cash_flow_data['fiscalDateEnding'],
                y=cash_flow_data['Free Cash Flow'],
                mode='lines+markers',
                line=dict(color='orange', width=2),
                marker=dict(symbol='circle', size=10)
            ))

            # Style the layout
            fig_cash_flow.update_layout(
            title=dict(text='Cash Flow', y=1, x=0, pad_l=20, pad_t=20, font_color='cornflowerblue', font_size=35),
            barmode='group',
            bargroupgap=0.2,
            barcornerradius=5,
            yaxis_title='Amount (in billions)',
            xaxis=dict(tickmode='linear'),  # Ensure each year is displayed
            margin=dict(l=100, r=30, t=60, b=40),
            height=400,  # Adjust the height of the chart
            paper_bgcolor=st.session_state.chart_bgcolor,  # Background color
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            showlegend=True,
            legend_font_color=st.session_state.text_color,
            legend=dict(x=0.45, xanchor='center', y=-0.1, orientation='h'),  # Legend at the bottom
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig_cash_flow, use_container_width=True, config={'displayModeBar': False})
