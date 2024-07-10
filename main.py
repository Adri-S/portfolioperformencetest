import streamlit as st
import os
import importlib.util
from hydralit_components import HyLoader, Loaders  # Überprüfen, ob der Import korrekt ist

st.set_page_config(
    page_title="Portfolio Performance",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = ["Your Portfolio", "KPIs", "Stock Searcher"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
urls = {}
styles = {
    "nav": {
        "background-color": "cornflowerblue",
        "justify-content": "center",
    },
    "span": {
        "color": "whitesmoke",
        "padding": "14px",
    },
    "active": {
        "background-color": "whitesmoke",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

# Navigation Bar
page = st.sidebar.radio("Navigation", pages)  # Einfachere Alternative ohne externe Bibliothek

# Dynamisch die Seitenmodule laden
def load_page_module(page_name):
    module_name = f"{page_name}"
    module_path = os.path.join(parent_dir, f"{page_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Zuordnen der Seitennamen zu Dateinamen
functions = {
    "Your Portfolio": "1_your_portfolio",
    "KPIs": "2_KPIs",
    "Stock Searcher": "3_stock_searcher",
}

# Ausgewählte Seite laden und ausführen
selected_page = functions.get(page)
if selected_page:
    page_module = load_page_module(selected_page)
    page_module.app()
