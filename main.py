import streamlit as st
import hydralit_components as hc
import datetime

# Importiere die separaten Seitenmodule
import home_page
import kpi_overview
import stocks_overview
import y_finance

# Make it look nice from the start
st.set_page_config(layout='wide')

# Specify the primary menu definition
menu_data = [
    # {'icon': "fa fa-home", 'label': "Home"},
    {'icon': "fa fa-plus", 'label': "KPI overview"},
    {'icon': "fa fa-line-chart", 'label': "Stocks overview"},
    {'icon': "fa fa-table", 'label': "Y-Finance"},
]

over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=False, # Will show the st hamburger as well as the navbar now!
    sticky_nav=True, # At the top or not
    sticky_mode='pinned', # Jumpy or not-jumpy, but sticky or pinned
)

# Add custom CSS to center headings and expand the navbar
st.markdown("""
    <style>
        .stApp {
            margin: 0;
            padding: 0;
        }
        nav {
            width: 100%;
            text-align: center;
        }
        .nav-item {
            display: inline-block;
            float: none;
        }
        .navbar-nav {
            display: inline-block;
            float: none;
        }
        .nav-link {
            display: inline-block !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Define actions based on menu item clicked
if menu_id == 'Home':
    home_page.display_home_page()  # Rufe die Funktion aus dem Modul auf
elif menu_id == 'KPI overview':
    kpi_overview.display_kpi_overview()  # Rufe die Funktion aus dem Modul auf
elif menu_id == 'Stocks overview':
    stocks_overview.display_stocks_overview()  # Rufe die Funktion aus dem Modul auf
elif menu_id == 'Y-Finance':
    y_finance.display_y_finance()  # Rufe die Funktion aus dem Modul auf
