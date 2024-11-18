import os
from pathlib import Path
import pandas as pd
import numpy as np
import dill
from PIL import Image
import streamlit as st
from catboost import CatBoostClassifier

from eclyon.transforms import process_df


path_to_repo = Path(__file__).parent.resolve()
path_to_data = path_to_repo / 'data' / 'clean_health.csv'



def display_bulldozer_price(index):
    # compute model prediction
    pred_price = st.session_state.model.predict([st.session_state.X.values[index]])[0]
    pred_price = int(pred_price)
    true_price = int(st.session_state.y[index])

    # display actual and predicted prices
    col_pred, col_gold = st.columns(2)
    with col_pred:
        st.subheader('estimated price')
        st.write(str(pred_price) + ' Euros')
    with col_gold:
        st.subheader('real price')
        st.write(str(true_price) + ' Euros')
    return


def display_bulldozer_features(index):
    st.subheader('Bulldozer features')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.values[index]
    for i, feature in enumerate(st.session_state.X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.info(feature)
            with val0:
                st.success(str(row[i]))
        elif ind == 1:
            with feat1:
                st.info(feature)
            with val1:
                st.success(str(row[i]))
    return


def init_session_state():
    # session state
    if 'loaded' not in st.session_state:
        # validation set given in notebook
        n_valid = 10000

        # import raw data
        df_raw = pd.read_csv(path_to_data)

        # preprocess data
        X, y, nas = process_df(df_raw, 'drinker')
        X, y = X[n_valid:], y[n_valid:]

        # load regression model
        path_to_model = path_to_repo / 'models'
        model = CatBoostClassifier().load_model(fname = (path_to_model / 'model_full').resolve())

       
        # store in cache
        st.session_state.loaded = True
        st.session_state.n_valid = n_valid
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.model = model
    


def app():
    init_session_state()
    st.title('Choose a bulldozer')
    options = ['-'] + list(range(1, st.session_state.n_valid + 1))
    index = st.selectbox(label = 'Choose a bulldozer index', options = options, index = 0)
    if index != '-':
        display_bulldozer_price(index)
        display_bulldozer_features(index)
    return


if __name__ == '__main__':
    app()