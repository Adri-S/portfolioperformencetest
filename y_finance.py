import streamlit as st

def display_y_finance():
    st.write("Y-Finance page content goes here.")
    st.markdown(
        """
        <a href="https://finance.yahoo.com/" target="_blank">
            <button style="background-color: #4CAF50; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Go to Yahoo Finance</button>
        </a>
        """,
        unsafe_allow_html=True
    )
