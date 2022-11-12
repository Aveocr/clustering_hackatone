import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
import plotly.express as px


def render():
    dataset_file = st.file_uploader('Upload dataset table', 'csv')
    if dataset_file is None:
        return

    dataset = pd.read_csv(dataset_file, sep=";")
    rent_okved = pd.read_csv('../data/otrasli.csv', index_col = None)
