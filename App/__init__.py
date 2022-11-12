import streamlit as st
from streamlit_option_menu import option_menu

from views.view import render as render_view
from views.model import render as render_model
from views.docs import render as render_docs


selected = option_menu(menu_title=None,
    options=["View", "Model", "Documentation"],
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


match selected:
    case 'View':
        render_view()
    case 'Model':
        render_model()
    case 'Documentation':
        render_docs()
