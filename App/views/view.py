import os
import math
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

    # business_by_region = dict()
    # regions = output['region'].unique()
    # for region in regions:
    #     business_by_region[region] = output['business_count'][output['region'] == region].median()

    # st.write(business_by_region)

    # df = pd.DataFrame()
    # labelencoder = LabelEncoder()

    with st.spinner('Предобработка данных'):
        dataset['okved'] = dataset['okved'].apply(lambda x : x.split('.')[0]).astype(int)
        output['okved'] = output['okved'].apply(lambda x : x.split('.')[0]).astype(int)

        test_batch = dataset[:100]

        # df["INN"] = dataset["INN"]
        # df["okved"] = dataset["okved"]
        # df["region"] = labelencoder.fit_transform(dataset["region"].values)
        # df["section"] = labelencoder.fit_transform(dataset["section"].values)
        # df["days"] = (datetime.now() - pd.to_datetime(dataset["registration_date"])).dt.days
        # df['business_count'] = 

        processed_batch = preprocess_batch(test_batch)
    
    with st.spinner('Смотрим признаки'):
        fig, ax = plt.subplots()

        # X = calc_batch_X(processed_batch)

        pca = PCA(n_components=2)
        X = pca.fit_transform(processed_batch.values)

        ax.scatter(X[:, 0], X[:, 1])

        st.header('Признаки')
        st.text(f"explained variance per feature {pca.explained_variance_ratio_}")
        st.text(f"2 compomemt - explained variance {np.sum(pca.explained_variance_ratio_):.3f}")
        st.pyplot(fig)

    with st.spinner('Смотрим признаки'):
        pca3 = PCA(n_components=3)                # оставим только два признака
        # новые признаки -- это линейная комбинация старых
        # они неинтерпретируемы, но позволяют оценить число кластеров
        X3 = pca3.fit_transform(processed_batch.values)

        fig = px.scatter_3d(x=X3[:,0], y=X3[:,1], z=X3[:,2], width=700)

        st.header('Трехмерное отображение признаков')
        st.text(f"3 compomemt - explained variance {np.sum(pca.explained_variance_ratio_):.3f}")
        st.plotly_chart(fig)

    with st.spinner('Отображаем кластеры'):
        kmeans = KMeans(init="k-means++", n_clusters=5)
        kmeans.fit(X)

        # # Step size of the mesh. Decrease to increase the quality of the VQ.
        # h = 100  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, (x_max - x_min) / 20),
            np.arange(y_min, y_max, (y_max - y_min) / 20)
        )

        # # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
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
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    st.header('Кластеры')
    st.pyplot(fig)

    output = pd.DataFrame()
    dataset_size = int(dataset.shape[0])
    # dataset_size = 100

    progress_bar = st.progress(0.0)

    for i in range(0, dataset_size, 100):
        batch = dataset[i:i+100]
        processed_batch = preprocess_batch(batch)
        X = calc_batch_X(processed_batch)
        prediction = predict_batch(batch, processed_batch, X)

        output = pd.concat([output, prediction])

        progress_bar.progress(i / dataset_size)

    st.dataframe(output)

    file_data = output.to_csv(sep=';', index=False).encode('utf-8')
    
    st.download_button(
        label='Скачать как CSV таблицу',
        data=file_data,
        file_name='dataset_output.csv',
        mime='text/csv',
    )


def preprocess_batch(batch: pd.DataFrame):
    df = pd.DataFrame()
    labelencoder = LabelEncoder()

    df["okved"] = batch["okved"]
    df["region"] = labelencoder.fit_transform(batch["region"].values)
    df["section"] = labelencoder.fit_transform(batch["section"].values)
    df["days"] = (datetime.now() - pd.to_datetime(batch["registration_date"])).dt.days

    return df


def calc_batch_X(batch: pd.DataFrame):
    pca = PCA(n_components=2)
    return pca.fit_transform(batch.values)


def predict_batch(source: pd.DataFrame, batch: pd.DataFrame, batch_X):
    kmeans = KMeans(init="k-means++", n_clusters=5)
    kmeans.fit(batch_X)

    x_min, x_max = batch_X[:, 0].min() - 1, batch_X[:, 0].max() + 1
    y_min, y_max = batch_X[:, 1].min() - 1, batch_X[:, 1].max() + 1

    step_x = (x_max - x_min) / 20
    step_y = (y_max - y_min) / 20

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_x),
        np.arange(y_min, y_max, step_y)
    )

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cluster_ids = []
    for x in batch_X:
        x_index = math.floor((x[0] - x_min) / step_x)
        y_index = math.floor((x[1] - y_min) / step_y)
        cluster_ids.append(int(Z[x_index][y_index]) + 1)

    output = pd.DataFrame()
    output['INN'] = source['INN']
    output['cluster_id'] = pd.Series(cluster_ids)

    return output
