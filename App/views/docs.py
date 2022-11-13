import streamlit as st

DOCS_STR = '''
Наша команда подготовила решения для данного кейса в рамках хакатона
[Цифровой прорыв 2022](https://hacks-ai.ru/hackathons/757130_).

- [Репозиторий проекта](https://github.com/Aveocr/clustering_hackatone)
'''

def render():
    st.markdown(DOCS_STR)
