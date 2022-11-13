import os
from datetime import datetime

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def render():
    with st.spinner('Загрузка датасета'):
        dataset_file = st.file_uploader('Upload dataset table', 'csv')
        if dataset_file is None:
            return

    with st.spinner('Загрузка таблиц'):
        dataset = pd.read_csv(dataset_file, sep=";")
        output = pd.read_csv('data/output_1.csv', encoding="cp1251", sep=";", index_col = False)

    df = pd.DataFrame()
    labelencoder = LabelEncoder()

    with st.spinner('Предобработка данных'):
        dataset['okved'] = dataset['okved'].apply(lambda x : x.split('.')[0]).astype(int)
        output['okved'] = output['okved'].apply(lambda x : x.split('.')[0]).astype(int)

        df["okved"] = output['okved']
        df["region"] = labelencoder.fit_transform(output["region"].values)
        df["section"] = labelencoder.fit_transform(output["section"].values)
        df["count"] = output["business_count"]
        df["day"] = (datetime.now() - pd.to_datetime(dataset["registration_date"])).dt.days
    
    with st.spinner('Смотрим признаки'):
        fig, ax = plt.subplots()

        pca = PCA(n_components=2)
        X = pca.fit_transform(df.values)

        ax.scatter(X[:, 0], X[:, 1])

        st.header('Признаки')
        st.text(f"explained variance per feature {pca.explained_variance_ratio_}")
        st.text(f"2 compomemt - explained variance {np.sum(pca.explained_variance_ratio_):.3f}")
        st.pyplot(fig)

    with st.spinner('Смотрим признаки'):
        pca3 = PCA(n_components=3)                # оставим только два признака
        # новые признаки -- это линейная комбинация старых
        # они неинтерпретируемы, но позволяют оценить число кластеров
        X3 = pca3.fit_transform( df.values )

        fig = px.scatter_3d( x=X3[:,0], y=X3[:,1], z=X3[:,2], width=700)

        st.header('Трехмерное отображение признаков')
        st.text(f"3 compomemt - explained variance {np.sum(pca.explained_variance_ratio_):.3f}")
        st.plotly_chart(fig)

    with st.spinner('Отображаем кластеры'):
        kmeans = KMeans(init="k-means++", n_clusters=5)
        kmeans.fit(X)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots()
        ax.figure(1)
        ax.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        ax.plot(X[:, 0], X[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        ax.xlim(x_min, x_max)
        ax.ylim(y_min, y_max)
        ax.xticks(())
        ax.yticks(())

    st.header('Кластеры')
    st.pyplot(fig)
