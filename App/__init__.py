import streamlit as st
from streamlit_option_menu import option_menu

selected = option_menu(menu_title=None,
    options=["view", "model", "documentation"],
    orientation="horizontal",
    icons=['cloud-upload', 'gear', 'list-task'],
    styles={
        "container": { "padding": "0!important", "background-color": "#1940FE" },
        "icon": { "color": "white", "font-size": "20px" },
        "nav-link": {
            "font-size": "20px",
            "text-align": "center",
            "color": "white",
            "margin":"0px",
            "--hover-color": "#4463fe"
        },
        "nav-link-selected": { "background-color": "blue" },
    }
)
