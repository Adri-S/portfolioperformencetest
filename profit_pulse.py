import streamlit as st

# Main content
st.title("Portfolio Performance Tracker")
st.write("""
**Welcome to the Portfolio Performance Tracker!**:money_with_wings:


The Portfolio Performance Tracker is designed to help you monitor and optimize your investment portfolio. With this app, you can:

- Track the performance of your stocks with detailed metrics.
- Analyze key performance indicators (KPIs) to understand your portfolio's health.
- Conduct Monte Carlo simulations to estimate future portfolio returns.
- Search and add new stocks to your portfolio seamlessly.
- Visualize your portfolio's performance with interactive charts and tables.

Stay on top of your investments and make informed decisions with the Portfolio Performance Tracker!
""")

# Initialize session state for the toggle if not already set
if 'darkmode' not in st.session_state:
    st.session_state.darkmode = False  # Default value (off)

if 'gauge_bgcolor' not in st.session_state:
    st.session_state.gauge_bgcolor = "whitesmoke"

darkmode = st.toggle("Activate Dark Mode", value=st.session_state.darkmode)

if darkmode != st.session_state.darkmode:
    st.session_state.darkmode = darkmode

st.write(f"The toggle is {'on' if st.session_state.darkmode else 'off'}")

if darkmode:
    st.session_state.background_color_page="#1E1E1E"
    st.session_state.chart_bgcolor="#282829"
    st.session_state.text_color="snow"
    st.session_state.gauge_bgcolor="#495057"
    st.session_state.border_color_input="#343a40"

else:
    st.session_state.background_color_page="snow"
    st.session_state.chart_bgcolor="#ffffff"
    st.session_state.text_color="black"
    st.session_state.gauge_bgcolor="whitesmoke"
    st.session_state.border_color_input="#e9ecef"